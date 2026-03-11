"""
全双工语音智能体模块

跨平台支持：Windows 和 macOS

架构：
- 音频输入线程：持续读取麦克风，处理 VAD
- TTS 播放线程：独立播放，不阻塞输入
- 主控制线程：处理对话逻辑
"""

import asyncio
import threading
import time
import logging
import platform
from collections import deque
from queue import Queue, Empty
from typing import Optional, Callable, Dict, Any
from enum import Enum

try:
    import pyaudio
    import numpy as np
except ImportError as e:
    raise ImportError("请安装依赖: pip install pyaudio numpy") from e

from .config import Config
from .state import AgentState
from .vad import VoiceActivityDetector, BargeInDetector
from .asr import SpeechRecognizer
from .llm import LanguageModel
from .tts import TextToSpeech
from .aec import SimpleAEC
from .interrupt import SemanticInterruptDetector

logger = logging.getLogger(__name__)

IS_WINDOWS = platform.system() == 'Windows'
IS_MACOS = platform.system() == 'Darwin'


class FullDuplexAgent:
    """全双工语音对话智能体"""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        
        # 组件
        self.vad = VoiceActivityDetector(self.config)
        self.asr = SpeechRecognizer(self.config)
        self.llm = LanguageModel(self.config)
        self.tts = TextToSpeech(self.config)
        
        # 回声消除
        self.aec = SimpleAEC(
            sample_rate=self.config.sample_rate,
            echo_suppression=0.8  # 80% 回声抑制
        )
        
        # PyAudio
        self._pyaudio_in: Optional[pyaudio.PyAudio] = None
        self._pyaudio_out: Optional[pyaudio.PyAudio] = None
        self._input_stream: Optional[pyaudio.Stream] = None
        
        # 音频队列（回调模式）
        self._audio_queue = Queue(maxsize=200)
        self._use_callback_mode = IS_MACOS  # macOS 使用回调模式
        
        # 状态
        self.state = AgentState.IDLE
        self._is_running = False
        self._state_lock = threading.Lock()
        
        # TTS 控制
        self._tts_playing = threading.Event()
        self._stop_playback = threading.Event()
        self._tts_start_time = 0
        self._tts_end_time = 0  # TTS 结束时间
        self._tts_thread: Optional[threading.Thread] = None
        self._tts_queue = Queue(maxsize=5)
        
        # TTS 参考信号缓冲（用于 AEC）
        self._tts_reference_buffer = deque(maxlen=16000 * 2)  # 2 秒缓冲
        self._tts_playback_position = 0  # 当前播放位置
        
        # 音频缓冲
        self._audio_buffer = bytearray()
        
        # 帧计数
        self._frame_count = 0
        
        # 线程
        self._input_thread: Optional[threading.Thread] = None
        self._tts_worker_thread: Optional[threading.Thread] = None

    def _set_state(self, new_state: AgentState):
        with self._state_lock:
            if self.state != new_state:
                logger.info(f"状态: {self.state.value} → {new_state.value}")
                self.state = new_state

    def _init_audio(self) -> bool:
        try:
            self._pyaudio_in = pyaudio.PyAudio()
            self._pyaudio_out = pyaudio.PyAudio()
            logger.info(f"✅ 音频设备初始化成功 ({platform.system()})")
            return True
        except Exception as e:
            logger.error(f"❌ 音频设备初始化失败: {e}")
            return False

    def _start_input_stream(self) -> bool:
        if self._input_stream is not None:
            return True
        
        try:
            if self._use_callback_mode:
                audio_queue = self._audio_queue
                
                def audio_callback(in_data, frame_count, time_info, status):
                    try:
                        audio_queue.put_nowait(in_data)
                    except:
                        pass
                    return (None, pyaudio.paContinue)
                
                self._input_stream = self._pyaudio_in.open(
                    rate=self.config.sample_rate,
                    channels=1,
                    format=pyaudio.paInt16,
                    input=True,
                    frames_per_buffer=480,
                    stream_callback=audio_callback
                )
                logger.info("🎤 麦克风输入流已启动 (回调模式)")
            else:
                self._input_stream = self._pyaudio_in.open(
                    rate=self.config.sample_rate,
                    channels=1,
                    format=pyaudio.paInt16,
                    input=True,
                    frames_per_buffer=480
                )
                logger.info("🎤 麦克风输入流已启动")
            
            return True
        except Exception as e:
            logger.error(f"❌ 启动麦克风失败: {e}")
            return False

    def _audio_input_loop(self):
        """音频输入主循环 - 不被 TTS 阻塞"""
        logger.info("🎧 音频输入线程启动")
        last_heartbeat = time.time()
        
        while self._is_running:
            try:
                if self._input_stream is None:
                    if not self._start_input_stream():
                        time.sleep(0.5)
                        continue
                
                # 获取音频数据
                try:
                    if self._use_callback_mode:
                        audio_data = self._audio_queue.get(timeout=0.1)
                    else:
                        audio_data = self._input_stream.read(480, exception_on_overflow=False)
                except Empty:
                    continue
                except Exception as e:
                    logger.debug(f"音频读取错误: {e}")
                    continue
                
                self._frame_count += 1
                
                # 心跳日志
                if time.time() - last_heartbeat > 1.0:
                    logger.info(f"[音频心跳] frame={self._frame_count}, tts_playing={self._tts_playing.is_set()}")
                    last_heartbeat = time.time()
                
                # 处理音频
                if self._tts_playing.is_set():
                    self._handle_interrupt_detection(audio_data)
                else:
                    self._handle_normal_vad(audio_data)
                    
            except Exception as e:
                logger.error(f"音频输入错误: {e}")
                time.sleep(0.1)
        
        logger.info("🎧 音频输入线程结束")

    def _handle_normal_vad(self, audio: bytes):
        """正常 VAD 处理"""
        # 检查是否在 TTS 播放后的抑制期内
        if self._tts_end_time > 0:
            elapsed = time.time() - self._tts_end_time
            if elapsed < 1.5:  # TTS 结束后 1.5 秒内抑制
                return
        
        result = self.vad.process(audio)
        
        if result.get('is_speech'):
            self._audio_buffer.extend(audio)
            if self.asr.is_connected:
                self.asr.send(audio)
            self._set_state(AgentState.LISTENING)
        
        elif result.get('speech_end') and len(self._audio_buffer) > 0:
            self._set_state(AgentState.THINKING)
            time.sleep(0.3)
            
            self.asr.stop()
            user_text = self.asr.get_result(timeout=0.5)
            self._audio_buffer = bytearray()
            self.vad.normal_vad.reset()
            
            if user_text and user_text.strip():
                logger.info(f"📝 用户: {user_text}")
                print(f"\n👤 你: {user_text}")
                
                response = self.llm.chat(user_text)
                if response:
                    print(f"🤖 助手: {response}")
                    # 将 TTS 任务放入队列，不阻塞
                    self._tts_queue.put(response)
            else:
                logger.warning("未识别到有效语音")
            
            self.asr.start()
            self._set_state(AgentState.IDLE)

    def _handle_interrupt_detection(self, audio: bytes):
        """打断检测处理 - 使用参考信号减法消除回声"""
        # 静默期：TTS 开始后 300ms 内不做检测
        if self._tts_start_time > 0:
            elapsed = time.time() - self._tts_start_time
            if elapsed < 0.3:
                return
        
        # 转换音频
        mic_samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32)
        
        # === 关键：参考信号减法 ===
        # 估计系统延迟（扬声器到麦克风的传播时间）
        # macOS 通常 20-100ms
        system_delay_ms = 50  # 可调整
        system_delay_samples = int(self.config.sample_rate * system_delay_ms / 1000)
        
        # 计算需要多少参考信号
        samples_needed = len(mic_samples) + system_delay_samples
        
        if len(self._tts_reference_buffer) >= samples_needed:
            # 获取对应的参考信号（考虑延迟）
            ref_samples = np.array(list(self._tts_reference_buffer)[-samples_needed:-system_delay_samples], dtype=np.float32)
            
            # 确保长度匹配
            if len(ref_samples) >= len(mic_samples):
                ref_samples = ref_samples[:len(mic_samples)]
                
                # 减去参考信号（回声消除）
                # 增益因子：麦克风捕获的回声通常比原始信号小
                echo_gain = 0.3  # 可调整
                clean_samples = mic_samples - ref_samples * echo_gain
                clean_samples = np.clip(clean_samples, -32768, 32767)
                
                # 转回 bytes
                clean_audio = clean_samples.astype(np.int16).tobytes()
            else:
                clean_audio = audio
        else:
            # 参考信号不足，使用原始信号
            clean_audio = audio
        
        # 计算处理后能量
        clean_energy = np.sqrt(np.mean(np.frombuffer(clean_audio, dtype=np.int16).astype(np.float32) ** 2))
        
        # 使用 Silero VAD 检测
        result = self.vad.process_for_interrupt(clean_audio)
        speech_prob = result.get('speech_prob', 0)
        is_speech = result.get('is_speech', False)
        
        # 只发送真正的语音给 ASR
        if is_speech and speech_prob > 0.6:
            if self.asr.is_connected:
                self.asr.send(clean_audio)
        
        # 打断条件：概率 > 0.75（提高阈值）
        if is_speech and speech_prob > 0.75:
            logger.debug(f"[打断检测] prob={speech_prob:.2f}, clean_energy={clean_energy:.1f}")
            
            if result.get('speech_start'):
                logger.info(f"⚡ 打断触发！prob={speech_prob:.2f}, energy={clean_energy:.1f}")
                self._stop_playback.set()

    def _tts_worker_loop(self):
        """TTS 播放工作线程"""
        logger.info("🔊 TTS 工作线程启动")
        
        while self._is_running:
            try:
                # 获取 TTS 任务
                text = self._tts_queue.get(timeout=0.5)
                if not text:
                    continue
                
                self._play_tts(text)
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"TTS 错误: {e}")
        
        logger.info("🔊 TTS 工作线程结束")

    def _play_tts(self, text: str):
        """播放 TTS"""
        audio = self.tts.synthesize(text)
        if not audio:
            return
        
        if len(audio) > 44 and audio[:4] == b'RIFF':
            audio = audio[44:]
        
        logger.info(f"🔊 播放 TTS ({len(audio)} 字节)")
        
        self._set_state(AgentState.SPEAKING)
        self._tts_playing.set()
        self._tts_start_time = time.time()
        self._stop_playback.clear()
        self.vad.set_tts_playing(True)
        self.vad.barge_in_detector.set_tts_state(True)
        
        # === 清空并准备参考信号缓冲 ===
        self._tts_reference_buffer.clear()
        
        # 重采样 TTS 音频到麦克风采样率（用于参考信号）
        # TTS 采样率: 22050Hz, 麦克风: 16000Hz
        tts_samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32)
        
        # 计算重采样比例
        resample_ratio = self.config.sample_rate / self.config.tts_sample_rate
        new_length = int(len(tts_samples) * resample_ratio)
        
        # 简单线性插值重采样
        original_indices = np.linspace(0, len(tts_samples) - 1, new_length)
        resampled = np.interp(original_indices, np.arange(len(tts_samples)), tts_samples)
        
        # 转换为 int16 并记录到参考缓冲
        resampled_int16 = resampled.astype(np.int16)
        
        # 启动 ASR
        if not self.asr.is_connected:
            self.asr.start()
        
        try:
            stream = self._pyaudio_out.open(
                rate=self.config.tts_sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                output=True,
                frames_per_buffer=1024
            )
            
            chunk_size = 1024
            interrupted = False
            
            # 计算每个输出 chunk 对应的重采样参考信号
            tts_chunk_samples = chunk_size  # TTS 采样
            ref_chunk_samples = int(tts_chunk_samples * resample_ratio)  # 对应的参考信号采样
            
            for i in range(0, len(audio), chunk_size):
                if self._stop_playback.is_set():
                    logger.info("⚡ TTS 被打断")
                    interrupted = True
                    break
                
                chunk = audio[i:i + chunk_size]
                stream.write(chunk, exception_on_underflow=False)
                
                # 记录对应的参考信号
                ref_start = int(i * resample_ratio)
                ref_end = ref_start + ref_chunk_samples
                if ref_end <= len(resampled_int16):
                    ref_chunk = resampled_int16[ref_start:ref_end]
                    for sample in ref_chunk:
                        self._tts_reference_buffer.append(float(sample))
            
            stream.stop_stream()
            stream.close()
            
            if interrupted:
                self._handle_interrupt()
            
        except Exception as e:
            logger.error(f"播放错误: {e}")
        finally:
            self._tts_playing.clear()
            self._tts_start_time = 0
            self._tts_end_time = time.time()  # 记录结束时间
            self._stop_playback.clear()
            self.vad.set_tts_playing(False)
            self.vad.barge_in_detector.set_tts_state(False)
            self._set_state(AgentState.IDLE)

    def _handle_interrupt(self):
        """处理打断"""
        logger.info("处理打断...")
        time.sleep(0.5)
        
        self.asr.stop()
        user_text = self.asr.get_result(timeout=0.3)
        if not user_text:
            user_text = self.asr.get_partial_text()
        
        if user_text and user_text.strip():
            logger.info(f"📝 打断: {user_text}")
            print(f"\n⚡ 你: {user_text}")
            
            self.asr.start()
            response = self.llm.chat(user_text)
            if response:
                print(f"🤖 助手: {response}")
                self._tts_queue.put(response)
        else:
            self.asr.start()

    async def run(self):
        """运行智能体"""
        print("=" * 60)
        print(f"🎙️ 全双工语音对话智能体 v2.3")
        print(f"🖥️ 平台: {platform.system()}")
        print("=" * 60)
        print(f"📦 模型: {self.config.llm_model}")
        print(f"🎤 VAD: {self.vad.vad_name}")
        print(f"⚡ 打断检测: 启用")
        print("-" * 60)
        print("💡 提示:")
        print("   - 说话后停顿 0.5 秒自动提交")
        print("   - TTS 播放时可说话打断")
        print("-" * 60)
        print("🛑 Ctrl+C 退出")
        print("=" * 60)
        
        if not self._init_audio():
            print("❌ 音频初始化失败")
            return
        
        if not self.asr.start():
            print("❌ ASR 启动失败")
            return
        
        self._is_running = True
        self._set_state(AgentState.IDLE)
        
        if not self._start_input_stream():
            print("❌ 麦克风启动失败")
            return
        
        # 启动线程
        self._input_thread = threading.Thread(target=self._audio_input_loop, daemon=True)
        self._input_thread.start()
        
        self._tts_worker_thread = threading.Thread(target=self._tts_worker_loop, daemon=True)
        self._tts_worker_thread.start()
        
        try:
            while self._is_running:
                await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
        finally:
            self._cleanup()

    def _cleanup(self):
        self._is_running = False
        
        if self._input_stream:
            try:
                self._input_stream.stop_stream()
                self._input_stream.close()
            except:
                pass
        
        self.asr.stop()
        
        if self._pyaudio_in:
            self._pyaudio_in.terminate()
        if self._pyaudio_out:
            self._pyaudio_out.terminate()