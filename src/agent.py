"""
全双工语音对话智能体

架构：Silero VAD → Paraformer → Qwen3-Omni → CosyVoice

核心规则：
1. 实时响应：静音≥0.3秒立即回复
2. 边听边说：TTS播放时持续监听，用户说话立即打断
3. 节奏自然：模拟真人打电话体验
"""

import asyncio
import threading
import time
import logging
import platform
from collections import deque
from queue import Queue, Empty
from typing import Optional, List
import re

try:
    import pyaudio
    import numpy as np
except ImportError as e:
    raise ImportError("请安装依赖: pip install pyaudio numpy") from e

from .config import Config
from .state import AgentState
from .vad import VoiceActivityDetector
from .asr import SpeechRecognizer
from .llm import LanguageModel
from .tts import TextToSpeech

logger = logging.getLogger(__name__)

IS_MACOS = platform.system() == 'Darwin'


class FullDuplexAgent:
    """全双工语音对话智能体"""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        
        # 组件初始化
        self.vad = VoiceActivityDetector(self.config)
        self.asr = SpeechRecognizer(self.config)
        self.llm = LanguageModel(self.config)
        self.tts = TextToSpeech(self.config)
        
        # PyAudio
        self._pyaudio_in: Optional[pyaudio.PyAudio] = None
        self._pyaudio_out: Optional[pyaudio.PyAudio] = None
        self._input_stream: Optional[pyaudio.Stream] = None
        
        # 音频队列（macOS 回调模式）
        self._audio_queue = Queue(maxsize=200)
        self._use_callback_mode = IS_MACOS
        
        # 状态
        self.state = AgentState.IDLE
        self._is_running = False
        self._state_lock = threading.Lock()
        
        # TTS 控制
        self._tts_playing = threading.Event()
        self._stop_playback = threading.Event()
        self._tts_thread: Optional[threading.Thread] = None
        self._tts_queue = Queue(maxsize=3)
        
        # 音频缓冲
        self._audio_buffer = bytearray()
        
        # AEC 参考 signal 缓冲
        self._aec_buffer = deque(maxlen=16000 * 3)  # 3秒缓冲
        self._aec_position = 0  # 当前播放位置
        
        # 上下文管理 - 最近3轮有效对话
        self._context: List[dict] = []
        self._max_context = 3
        
        # 帧计数
        self._frame_count = 0
        self._min_silence_ms = 300  # 0.3秒静音触发回复
        
        # 打断检测状态
        self._interrupt_speech_frames = 0
        self._interrupt_threshold = 5
        self._tts_frame_count = 0
        self._tts_grace_period = 20  # 20帧静默期
        self._interrupt_prob_threshold = 0.7
        
        # 线程
        self._input_thread: Optional[threading.Thread] = None

    def _set_state(self, new_state: AgentState):
        with self._state_lock:
            if self.state != new_state:
                logger.info(f"状态: {self.state.value} → {new_state.value}")
                self.state = new_state

    def _init_audio(self) -> bool:
        try:
            self._pyaudio_in = pyaudio.PyAudio()
            self._pyaudio_out = pyaudio.PyAudio()
            logger.info(f"✅ 音频初始化成功 ({platform.system()})")
            return True
        except Exception as e:
            logger.error(f"❌ 音频初始化失败: {e}")
            return False

    def _start_input_stream(self) -> bool:
        if self._input_stream is not None:
            return True
        
        try:
            if self._use_callback_mode:
                audio_queue = self._audio_queue
                
                def callback(in_data, frame_count, time_info, status):
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
                    stream_callback=callback
                )
                logger.info("🎤 输入流启动 (回调模式)")
            else:
                self._input_stream = self._pyaudio_in.open(
                    rate=self.config.sample_rate,
                    channels=1,
                    format=pyaudio.paInt16,
                    input=True,
                    frames_per_buffer=480
                )
                logger.info("🎤 输入流启动")
            return True
        except Exception as e:
            logger.error(f"❌ 麦克风启动失败: {e}")
            return False

    def _audio_input_loop(self):
        """音频输入主循环"""
        logger.info("🎧 音频输入线程启动")
        last_heartbeat = time.time()
        
        while self._is_running:
            try:
                if self._input_stream is None:
                    if not self._start_input_stream():
                        time.sleep(0.5)
                        continue
                
                # 获取音频
                try:
                    if self._use_callback_mode:
                        audio = self._audio_queue.get(timeout=0.1)
                    else:
                        audio = self._input_stream.read(480, exception_on_overflow=False)
                except Empty:
                    continue
                except Exception as e:
                    continue
                
                self._frame_count += 1
                
                # 心跳
                if time.time() - last_heartbeat > 3.0:
                    tts_state = self._tts_playing.is_set()
                    logger.info(f"[心跳] frame={self._frame_count}, tts={tts_state}")
                    last_heartbeat = time.time()
                
                # 核心逻辑
                if self._tts_playing.is_set():
                    # TTS 播放中：检测打断
                    self._detect_interrupt(audio)
                else:
                    # 正常语音检测
                    self._handle_vad(audio)
                    
            except Exception as e:
                logger.error(f"音频错误: {e}")
                time.sleep(0.05)
        
        logger.info("🎧 输入线程结束")

    def _detect_interrupt(self, audio: bytes):
        """
        打断检测 - 使用 AEC 回声消除
        
        核心原理：
        1. 从麦克风输入减去 TTS 参考信号
        2. 用 Silero VAD 检测消除回声后的信号
        """
        self._tts_frame_count += 1
        
        # 静默期
        if self._tts_frame_count <= self._tts_grace_period:
            return
        
        # === AEC 处理 ===
        mic_samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32)
        
        # 从参考缓冲获取对应的样本
        if len(self._aec_buffer) >= len(mic_samples):
            # 计算延迟对齐（麦克风采样点对应之前的参考信号）
            delay_samples = int(16000 * 0.05)  # 50ms 延迟
            start_idx = max(0, len(self._aec_buffer) - len(mic_samples) - delay_samples)
            end_idx = start_idx + len(mic_samples)
            
            if end_idx <= len(self._aec_buffer):
                ref_samples = np.array(list(self._aec_buffer)[start_idx:end_idx], dtype=np.float32)
                
                # 减去参考信号（回声消除）
                # 回声增益：麦克风捕获的回声通常比原始信号小
                echo_gain = 0.4
                clean_samples = mic_samples - ref_samples * echo_gain
                clean_samples = np.clip(clean_samples, -32768, 32767)
                clean_audio = clean_samples.astype(np.int16).tobytes()
            else:
                clean_audio = audio
        else:
            clean_audio = audio
        
        # 使用 Silero VAD 检测消除回声后的信号
        vad_result = self.vad.normal_vad.process(clean_audio)
        
        speech_prob = vad_result.get('speech_prob', 0)
        is_speech = vad_result.get('is_speech', False)
        
        # 调试日志
        if speech_prob > 0.3:
            logger.info(f"[打断] prob={speech_prob:.2f}, frames={self._interrupt_speech_frames}")
        
        # 条件
        if is_speech and speech_prob > self._interrupt_prob_threshold:
            self._interrupt_speech_frames += 1
            
            # 发送给 ASR
            if self.asr.is_connected:
                self.asr.send(clean_audio)
            
            if self._interrupt_speech_frames >= self._interrupt_threshold:
                logger.info(f"⚡ 打断！prob={speech_prob:.2f}")
                self._stop_playback.set()
        else:
            self._interrupt_speech_frames = max(0, self._interrupt_speech_frames - 1)

    def _handle_vad(self, audio: bytes):
        """正常 VAD 处理"""
        result = self.vad.process(audio)
        
        if result.get('is_speech'):
            self._audio_buffer.extend(audio)
            if self.asr.is_connected:
                self.asr.send(audio)
            self._set_state(AgentState.LISTENING)
        
        elif result.get('speech_end') and len(self._audio_buffer) > 0:
            self._set_state(AgentState.THINKING)
            
            # 等待 ASR 完成
            time.sleep(0.3)
            self.asr.stop()
            
            user_text = self.asr.get_result(timeout=0.5)
            
            # 清空缓冲
            self._audio_buffer = bytearray()
            self.vad.normal_vad.reset()
            
            if user_text:
                user_text = self._clean_asr_text(user_text)
            
            if user_text:
                logger.info(f"📝 用户: {user_text}")
                print(f"\n👤 你: {user_text}")
                
                # 生成回复
                response = self._generate_response(user_text)
                
                if response:
                    print(f"🤖 助手: {response}")
                    self._tts_queue.put(response)
            else:
                # 兜底
                self._tts_queue.put("我没太听清，可以再说一遍吗？")
            
            self.asr.start()
            self._set_state(AgentState.IDLE)

    def _clean_asr_text(self, text: str) -> str:
        """清理 ASR 文本"""
        if not text:
            return ""
        
        text = text.strip()
        
        # 移除 [模糊] 等标注
        text = re.sub(r'\[.*?\]', '', text)
        
        # 过滤纯语气词
        filler_only = re.match(r'^[嗯呃啊哦额唔]+$', text)
        if filler_only:
            return ""
        
        # 移除开头结尾语气词
        text = re.sub(r'^[嗯呃啊哦额]+', '', text)
        text = re.sub(r'[嗯呃啊哦额]+$', '', text)
        
        return text.strip()

    def _generate_response(self, user_text: str) -> str:
        """生成回复"""
        # 添加到上下文
        self._context.append({"role": "user", "content": user_text})
        if len(self._context) > self._max_context * 2:
            self._context = self._context[-self._max_context * 2:]
        
        # 生成
        response = self.llm.chat(user_text)
        
        if response:
            # 口语化处理
            response = self._format_for_tts(response)
            self._context.append({"role": "assistant", "content": response})
        
        return response

    def _format_for_tts(self, text: str) -> str:
        """格式化为 TTS 口语文本"""
        if not text:
            return ""
        
        # 移除 markdown
        text = re.sub(r'\*+([^*]+)\*+', r'\1', text)
        text = re.sub(r'`+([^`]+)`+', r'\1', text)
        text = re.sub(r'#+ ', '', text)
        
        # 仅保留逗号句号
        text = re.sub(r'[，。、；：！？,.!?;:]', lambda m: '，' if m.group() in '，,;' else '。', text)
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。]', '', text)
        
        # 长度控制
        if len(text) > 80:
            sentences = re.split(r'[。]', text)
            result = ""
            for s in sentences:
                if len(result + s) <= 80:
                    result += s + "。"
                else:
                    break
            text = result.rstrip("。")
        
        return text.strip()

    def _tts_loop(self):
        """TTS 播放循环"""
        logger.info("🔊 TTS 线程启动")
        
        while self._is_running:
            try:
                text = self._tts_queue.get(timeout=0.5)
                if text:
                    self._play_tts(text)
            except Empty:
                continue
            except Exception as e:
                logger.error(f"TTS 错误: {e}")
        
        logger.info("🔊 TTS 线程结束")

    def _play_tts(self, text: str):
        """播放 TTS"""
        audio = self.tts.synthesize(text)
        if not audio:
            return
        
        # 跳过 WAV 头
        if len(audio) > 44 and audio[:4] == b'RIFF':
            audio = audio[44:]
        
        logger.info(f"🔊 播放 ({len(audio)} 字节): {text[:30]}...")
        
        # 设置状态
        self._tts_playing.set()
        self._set_state(AgentState.SPEAKING)
        
        # 重置打断检测状态
        self._tts_frame_count = 0
        self._interrupt_speech_frames = 0
        self._stop_playback.clear()
        
        # === AEC: 准备参考信号 ===
        self._aec_buffer.clear()
        
        # 重采样 TTS 音频到麦克风采样率 (22050Hz -> 16000Hz)
        tts_samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32)
        resample_ratio = 16000 / 22050
        new_length = int(len(tts_samples) * resample_ratio)
        original_indices = np.linspace(0, len(tts_samples) - 1, new_length)
        resampled = np.interp(original_indices, np.arange(len(tts_samples)), tts_samples)
        resampled_int16 = resampled.astype(np.int16)
        
        # 存入参考缓冲
        for s in resampled_int16:
            self._aec_buffer.append(float(s))
        
        # 确保 ASR 运行
        if not self.asr.is_connected:
            self.asr.start()
        
        interrupted = False
        
        try:
            stream = self._pyaudio_out.open(
                rate=self.config.tts_sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                output=True,
                frames_per_buffer=1024
            )
            
            chunk_size = 1024
            for i in range(0, len(audio), chunk_size):
                if self._stop_playback.is_set():
                    logger.info("⚡ TTS 被打断")
                    interrupted = True
                    break
                stream.write(audio[i:i+chunk_size])
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            logger.error(f"播放错误: {e}")
        finally:
            self._tts_playing.clear()
            self._stop_playback.clear()
            self._set_state(AgentState.IDLE)
            
            if interrupted:
                self._handle_interrupt()

    def _handle_interrupt(self):
        """处理打断"""
        logger.info("处理打断...")
        
        # 清除打断标志
        self._stop_playback.clear()
        
        # 等待 ASR 完成
        time.sleep(0.5)
        self.asr.stop()
        time.sleep(0.2)
        
        # 获取新识别的结果
        user_text = self.asr.get_result(timeout=0.3)
        
        # 清空 partial text，避免下次误用
        self.asr.clear()
        
        if user_text and user_text.strip():
            user_text = self._clean_asr_text(user_text)
        
        if user_text:
            logger.info(f"📝 打断识别: {user_text}")
            print(f"\n⚡ 你: {user_text}")
            
            self.asr.start()
            response = self._generate_response(user_text)
            
            if response:
                print(f"🤖 助手: {response}")
                self._tts_queue.put(response)
        else:
            logger.info("打断但未识别到新语音")
            self.asr.start()

    async def run(self):
        """运行"""
        print("=" * 60)
        print("🎙️ 全双工语音助手")
        print("=" * 60)
        print(f"模型: {self.config.llm_model}")
        print(f"VAD: Silero")
        print("-" * 60)
        print("💡 说话后停顿 0.3 秒自动回复")
        print("💡 播放时说话可打断")
        print("-" * 60)
        print("Ctrl+C 退出")
        print("=" * 60)
        
        if not self._init_audio():
            return
        
        if not self.asr.start():
            print("❌ ASR 启动失败")
            return
        
        self._is_running = True
        self._set_state(AgentState.IDLE)
        
        if not self._start_input_stream():
            return
        
        # 启动线程
        self._input_thread = threading.Thread(target=self._audio_input_loop, daemon=True)
        self._input_thread.start()
        
        self._tts_thread = threading.Thread(target=self._tts_loop, daemon=True)
        self._tts_thread.start()
        
        try:
            while self._is_running:
                await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            print("\n👋 再见！")
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