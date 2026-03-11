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
        
        # 上下文管理 - 最近3轮有效对话
        self._context: List[dict] = []
        self._max_context = 3
        
        # 帧计数
        self._frame_count = 0
        
        # 打断检测状态
        self._tts_energy_baseline = 0.0
        self._tts_frames = 0
        self._interrupt_frames = 0
        self._interrupt_threshold = 8  # 连续8帧(~240ms)确认打断
        self._min_silence_ms = 300  # 0.3秒静音触发回复
        self._tts_start_time = 0  # TTS开始时间
        self._silence_period = 0.5  # TTS开始后0.5秒内不检测打断
        
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
                    logger.info(f"[心跳] frame={self._frame_count}")
                    last_heartbeat = time.time()
                
                # TTS 播放时：打断检测
                if self._tts_playing.is_set():
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
        打断检测 - 基于能量突变
        
        关键改进：
        1. TTS开始后0.5秒静默期，不检测
        2. 静默期后建立能量基线
        3. 检测能量显著增加才触发
        """
        # 静默期检查
        if self._tts_start_time > 0:
            elapsed = time.time() - self._tts_start_time
            if elapsed < self._silence_period:
                # 静默期内，不检测
                return
        
        samples = np.frombuffer(audio, dtype=np.int16)
        energy = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
        
        # 基线建立阶段（静默期后20帧）
        if self._tts_frames < 20:
            self._tts_energy_baseline = energy
            self._tts_frames += 1
            return
        
        # 慢速更新基线
        alpha = 0.02  # 更慢的更新速度
        self._tts_energy_baseline = alpha * energy + (1-alpha) * self._tts_energy_baseline
        
        # 检测突变
        ratio = energy / self._tts_energy_baseline if self._tts_energy_baseline > 0 else 1.0
        increase = energy - self._tts_energy_baseline
        
        # 提高阈值：比率>1.8 且 增量>200
        if ratio > 1.8 and increase > 200:
            self._interrupt_frames += 1
            if self._interrupt_frames >= self._interrupt_threshold:
                logger.info(f"⚡ 打断: ratio={ratio:.2f}, increase={increase:.0f}")
                self._stop_playback.set()
                self._interrupt_frames = 0
        else:
            self._interrupt_frames = max(0, self._interrupt_frames - 1)

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
        """
        清理 ASR 文本
        
        - 过滤纯语气词
        - 处理 [模糊] 标注
        - 去除无效内容
        """
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
        # 检查打断
        if self._stop_playback.is_set():
            return ""
        
        # 添加到上下文
        self._context.append({"role": "user", "content": user_text})
        if len(self._context) > self._max_context * 2:
            self._context = self._context[-self._max_context * 2:]
        
        # 生成
        response = self.llm.chat(user_text)
        
        if self._stop_playback.is_set():
            return ""
        
        if response:
            # 口语化处理
            response = self._format_for_tts(response)
            self._context.append({"role": "assistant", "content": response})
        
        return response

    def _format_for_tts(self, text: str) -> str:
        """
        格式化为 TTS 口语文本
        
        规则：
        - 仅保留逗号句号
        - ≤80字
        - 口语化
        """
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
            # 在句号处截断
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
        
        # 状态
        self._set_state(AgentState.SPEAKING)
        self._tts_playing.set()
        self._stop_playback.clear()
        
        # 重置打断检测 - 关键：设置开始时间
        self._tts_start_time = time.time()
        self._tts_energy_baseline = 0
        self._tts_frames = 0
        self._interrupt_frames = 0
        
        # 确保 ASR 运行
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
            for i in range(0, len(audio), chunk_size):
                if self._stop_playback.is_set():
                    logger.info("⚡ 被打断")
                    break
                stream.write(audio[i:i+chunk_size])
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            logger.error(f"播放错误: {e}")
        finally:
            self._tts_playing.clear()
            self._tts_start_time = 0
            self._stop_playback.clear()
            self._set_state(AgentState.IDLE)

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