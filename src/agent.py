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
        self._tts_thread: Optional[threading.Thread] = None
        self._tts_queue = Queue(maxsize=5)
        
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
        """打断检测处理"""
        result = self.vad.process_for_interrupt(audio)
        
        if result.get('is_speech'):
            if self.asr.is_connected:
                self.asr.send(audio)
        
        if result.get('speech_start'):
            logger.info("⚡ 打断触发！")
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
        self._stop_playback.clear()
        self.vad.set_tts_playing(True)
        self.vad.barge_in_detector.set_tts_state(True)
        
        # 启动 ASR 支持打断
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
            
            for i in range(0, len(audio), chunk_size):
                if self._stop_playback.is_set():
                    logger.info("⚡ TTS 被打断")
                    interrupted = True
                    break
                
                chunk = audio[i:i + chunk_size]
                stream.write(chunk, exception_on_underflow=False)
            
            stream.stop_stream()
            stream.close()
            
            if interrupted:
                self._handle_interrupt()
            
        except Exception as e:
            logger.error(f"播放错误: {e}")
        finally:
            self._tts_playing.clear()
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