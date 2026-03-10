"""
全双工语音智能体模块

架构改进：
- 使用状态机清晰管理状态转换
- 分离音频输入/输出线程
- 改进打断检测和处理流程
- 添加健壮的错误恢复机制
"""

import asyncio
import threading
import time
import logging
from queue import Queue, Empty
from typing import Optional, Callable
from enum import Enum
from dataclasses import dataclass

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


@dataclass
class AudioChunk:
    """音频数据块"""
    data: bytes
    timestamp: float
    frame_num: int


class FullDuplexAgent:
    """
    全双工语音对话智能体
    
    架构：
    ┌─────────────────────────────────────────────────┐
    │                   主控制循环                      │
    ├─────────────────────────────────────────────────┤
    │  ┌──────────┐    ┌──────────┐    ┌──────────┐  │
    │  │ 音频输入  │───→│   VAD    │───→│   ASR    │  │
    │  │  线程    │    │          │    │          │  │
    │  └──────────┘    └────┬─────┘    └────┬─────┘  │
    │                       │               │        │
    │                       ▼               ▼        │
    │               ┌──────────┐    ┌──────────┐    │
    │               │ 打断检测  │    │   LLM   │    │
    │               │          │    │         │    │
    │               └────┬─────┘    └────┬────┘    │
    │                    │               │         │
    │                    ▼               ▼         │
    │              ┌──────────┐    ┌──────────┐   │
    │              │  TTS 控制 │←───│  输出队列 │   │
    │              └──────────┘    └──────────┘   │
    └─────────────────────────────────────────────────┘
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        
        # 初始化组件
        self.vad = VoiceActivityDetector(self.config)
        self.asr = SpeechRecognizer(self.config)
        self.llm = LanguageModel(self.config)
        self.tts = TextToSpeech(self.config)
        self.interrupt_detector = SemanticInterruptDetector(self.config)
        
        # PyAudio 实例
        self._pyaudio_in: Optional[pyaudio.PyAudio] = None
        self._pyaudio_out: Optional[pyaudio.PyAudio] = None
        self._input_stream: Optional[pyaudio.Stream] = None
        self._output_stream: Optional[pyaudio.Stream] = None
        
        # 状态
        self.state = AgentState.IDLE
        self._is_running = False
        self._state_lock = threading.Lock()
        
        # 音频缓冲
        self._audio_buffer = bytearray()
        self._interrupt_audio = bytearray()
        
        # 打断控制
        self._should_interrupt = threading.Event()
        self._tts_playing = threading.Event()
        self._stop_playback = threading.Event()
        
        # 线程
        self._input_thread: Optional[threading.Thread] = None
        self._process_thread: Optional[threading.Thread] = None
        
        # 帧计数
        self._frame_count = 0
        
        # 回调（可用于 UI 更新）
        self._on_state_change: Optional[Callable[[AgentState], None]] = None
        self._on_user_text: Optional[Callable[[str], None]] = None
        self._on_assistant_text: Optional[Callable[[str], None]] = None
    
    def set_callbacks(self, on_state_change=None, on_user_text=None, on_assistant_text=None):
        """设置回调函数"""
        self._on_state_change = on_state_change
        self._on_user_text = on_user_text
        self._on_assistant_text = on_assistant_text
    
    def _set_state(self, new_state: AgentState):
        """更新状态"""
        with self._state_lock:
            if self.state != new_state:
                old_state = self.state
                self.state = new_state
                logger.info(f"状态: {old_state.value} → {new_state.value}")
                if self._on_state_change:
                    self._on_state_change(new_state)
    
    def _init_audio(self):
        """初始化音频设备"""
        try:
            self._pyaudio_in = pyaudio.PyAudio()
            self._pyaudio_out = pyaudio.PyAudio()
            logger.info("✅ 音频设备初始化成功")
            return True
        except Exception as e:
            logger.error(f"❌ 音频设备初始化失败: {e}")
            return False
    
    def _start_input_stream(self):
        """启动麦克风输入流"""
        if self._input_stream is not None:
            return True
        
        try:
            self._input_stream = self._pyaudio_in.open(
                rate=self.config.sample_rate,
                channels=self.config.channels,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=int(self.config.sample_rate * 0.03),  # 30ms
                stream_callback=None
            )
            logger.info("🎤 麦克风输入流已启动")
            return True
        except Exception as e:
            logger.error(f"❌ 启动麦克风失败: {e}")
            return False
    
    def _stop_input_stream(self):
        """停止麦克风输入流"""
        if self._input_stream:
            try:
                self._input_stream.stop_stream()
                self._input_stream.close()
            except:
                pass
            self._input_stream = None
            logger.info("🎤 麦克风输入流已停止")
    
    def _audio_input_loop(self):
        """音频输入主循环"""
        logger.info("🎧 音频输入线程启动")
        
        while self._is_running:
            try:
                # 确保输入流存在
                if self._input_stream is None:
                    if not self._start_input_stream():
                        time.sleep(0.5)
                        continue
                
                # 读取音频
                try:
                    chunk_size = int(self.config.sample_rate * 0.03)  # 30ms
                    audio_data = self._input_stream.read(chunk_size, exception_on_overflow=False)
                except OSError as e:
                    if "Stream closed" in str(e) or "Invalid stream" in str(e):
                        self._input_stream = None
                        time.sleep(0.1)
                        continue
                    raise
                
                self._frame_count += 1
                
                # 根据当前状态处理
                if self._tts_playing.is_set():
                    # TTS 播放中 - 打断检测模式
                    self._handle_interrupt_detection(audio_data)
                else:
                    # 正常模式
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
            # 语音结束，处理对话
            self._set_state(AgentState.THINKING)
            time.sleep(0.3)  # 等待 ASR 完成
            
            self.asr.stop()
            self._process_user_input()
            self.asr.start()
            
            # 重置
            self._audio_buffer = bytearray()
            self.vad.normal_vad.reset()
    
    def _handle_interrupt_detection(self, audio: bytes):
        """打断检测处理"""
        result = self.vad.process_for_interrupt(audio)
        
        # 调试日志
        if self._frame_count % 20 == 0 or result.get('speech_start'):
            logger.debug(
                f"[打断检测] frame={self._frame_count}, "
                f"energy={result.get('energy', 0):.1f}, "
                f"baseline={result.get('baseline', 0):.1f}, "
                f"ratio={result.get('energy_ratio', 0):.2f}, "
                f"is_speech={result.get('is_speech')}"
            )
        
        # 检测到语音
        if result.get('is_speech'):
            self._interrupt_audio.extend(audio)
            if self.asr.is_connected:
                self.asr.send(audio)
        
        # 触发打断
        if result.get('speech_start'):
            logger.info(f"⚡ 打断触发！")
            self._should_interrupt.set()
            self._stop_playback.set()
    
    def _process_user_input(self):
        """处理用户输入"""
        user_text = self.asr.get_result(timeout=0.5)
        
        if not user_text or not user_text.strip():
            logger.warning("未识别到有效语音")
            self._set_state(AgentState.IDLE)
            return
        
        logger.info(f"📝 用户: {user_text}")
        print(f"\n👤 你: {user_text}")
        
        if self._on_user_text:
            self._on_user_text(user_text)
        
        # 获取回复
        self._set_state(AgentState.THINKING)
        response = self.llm.chat(user_text)
        
        if response:
            print(f"🤖 助手: {response}")
            if self._on_assistant_text:
                self._on_assistant_text(response)
            self._play_response(response)
        else:
            self._set_state(AgentState.IDLE)
        
        self.asr.clear()
    
    def _play_response(self, text: str):
        """播放 TTS 响应"""
        # 合成语音
        audio = self.tts.synthesize(text)
        if not audio:
            self._set_state(AgentState.IDLE)
            return
        
        # 跳过 WAV 头
        if len(audio) > 44 and audio[:4] == b'RIFF':
            audio = audio[44:]
        
        logger.info(f"🔊 播放 TTS ({len(audio)} 字节)")
        
        # 设置状态
        self._set_state(AgentState.SPEAKING)
        self._tts_playing.set()
        self._stop_playback.clear()
        self._should_interrupt.clear()
        self._interrupt_audio = bytearray()
        self.vad.set_tts_playing(True)
        
        # 初始化打断检测器
        self.vad.barge_in_detector.set_tts_state(True)
        
        interrupted = False
        
        try:
            # 创建输出流
            stream = self._pyaudio_out.open(
                rate=self.config.tts_sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                output=True,
                frames_per_buffer=1024
            )
            
            # 播放
            chunk_size = 1024
            for i in range(0, len(audio), chunk_size):
                # 检查打断
                if self._stop_playback.is_set():
                    logger.info("⚡ TTS 被打断")
                    interrupted = True
                    break
                
                chunk = audio[i:i + chunk_size]
                stream.write(chunk, exception_on_underflow=False)
            
            # 清理
            stream.stop_stream()
            stream.close()
        
        except Exception as e:
            logger.error(f"播放错误: {e}")
        
        finally:
            # 重置状态
            self._tts_playing.clear()
            self._stop_playback.clear()
            self.vad.set_tts_playing(False)
            self.vad.barge_in_detector.set_tts_state(False)
            
            if interrupted:
                self._handle_interrupt()
            else:
                self._set_state(AgentState.IDLE)
    
    def _handle_interrupt(self):
        """处理打断"""
        logger.info("处理打断...")
        self._should_interrupt.clear()
        
        # 等待 ASR 完成
        time.sleep(0.5)
        self.asr.stop()
        time.sleep(0.2)
        
        # 获取识别结果
        user_text = self.asr.get_result(timeout=0.3)
        if not user_text:
            user_text = self.asr.get_partial_text()
        
        self._interrupt_audio = bytearray()
        
        if user_text and user_text.strip():
            logger.info(f"📝 打断识别: {user_text}")
            print(f"\n⚡ 你: {user_text}")
            
            self.asr.start()
            self._set_state(AgentState.THINKING)
            
            response = self.llm.chat(user_text)
            print(f"🤖 助手: {response}")
            
            if response:
                self._play_response(response)
            else:
                self._set_state(AgentState.IDLE)
        else:
            logger.info("打断但未识别到语音")
            self.asr.start()
            self._set_state(AgentState.IDLE)
    
    async def run(self):
        """运行智能体"""
        print("=" * 60)
        print("🎙️  全双工语音对话智能体 v2.1")
        print("=" * 60)
        print(f"📦 模型: {self.config.llm_model}")
        print(f"🎵 音色: {self.config.tts_voice}")
        print(f"⚡ 打断检测: {'启用' if self.config.barge_in_enabled else '禁用'}")
        print("-" * 60)
        print("💡 提示:")
        print("   - 说话后停顿 0.5 秒自动提交")
        print("   - TTS 播放时可说话打断")
        print("   - 打断时需要说话声音足够大")
        print("-" * 60)
        print("🛑 Ctrl+C 退出")
        print("=" * 60)
        
        # 初始化
        if not self._init_audio():
            print("❌ 音频设备初始化失败")
            return
        
        # 启动 ASR
        if not self.asr.start():
            print("❌ ASR 启动失败")
            return
        
        self._is_running = True
        self._set_state(AgentState.IDLE)
        
        # 启动麦克风
        if not self._start_input_stream():
            print("❌ 麦克风启动失败")
            return
        
        # 启动音频输入线程
        self._input_thread = threading.Thread(target=self._audio_input_loop, daemon=True)
        self._input_thread.start()
        
        # 主循环
        try:
            while self._is_running:
                await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """清理资源"""
        logger.info("清理资源...")
        self._is_running = False
        self._stop_input_stream()
        self.asr.stop()
        
        if self._pyaudio_in:
            self._pyaudio_in.terminate()
        if self._pyaudio_out:
            self._pyaudio_out.terminate()
        
        logger.info("资源清理完成")
    
    def stop(self):
        """停止智能体"""
        self._is_running = False
        self._stop_playback.set()
    
    @property
    def is_running(self) -> bool:
        return self._is_running