"""
全双工语音智能体模块

跨平台支持：Windows 和 macOS

架构改进：
- 使用状态机清晰管理状态转换
- 分离音频输入/输出线程
- 改进打断检测和处理流程
- 添加健壮的错误恢复机制
- 跨平台音频设备管理
"""

import asyncio
import threading
import time
import logging
import platform
import sys
from pathlib import Path
from queue import Queue, Empty
from typing import Optional, Callable, List, Dict, Any
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

# 平台检测
IS_WINDOWS = platform.system() == 'Windows'
IS_MACOS = platform.system() == 'Darwin'


@dataclass
class AudioDevice:
    """音频设备信息"""
    index: int
    name: str
    channels: int
    sample_rate: int
    is_input: bool


class AudioDeviceManager:
    """
    跨平台音频设备管理器
    
    处理 Windows 和 macOS 的音频设备差异
    """
    
    def __init__(self, pyaudio_instance: pyaudio.PyAudio):
        self._pyaudio = pyaudio_instance
    
    def list_input_devices(self) -> List[AudioDevice]:
        """列出所有输入设备"""
        devices = []
        for i in range(self._pyaudio.get_device_count()):
            info = self._pyaudio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                devices.append(AudioDevice(
                    index=i,
                    name=info['name'],
                    channels=info['maxInputChannels'],
                    sample_rate=int(info['defaultSampleRate']),
                    is_input=True
                ))
        return devices
    
    def list_output_devices(self) -> List[AudioDevice]:
        """列出所有输出设备"""
        devices = []
        for i in range(self._pyaudio.get_device_count()):
            info = self._pyaudio.get_device_info_by_index(i)
            if info['maxOutputChannels'] > 0:
                devices.append(AudioDevice(
                    index=i,
                    name=info['name'],
                    channels=info['maxOutputChannels'],
                    sample_rate=int(info['defaultSampleRate']),
                    is_input=False
                ))
        return devices
    
    def get_default_input_device(self) -> Optional[AudioDevice]:
        """获取默认输入设备"""
        try:
            info = self._pyaudio.get_default_input_device_info()
            return AudioDevice(
                index=info['index'],
                name=info['name'],
                channels=info['maxInputChannels'],
                sample_rate=int(info['defaultSampleRate']),
                is_input=True
            )
        except:
            return None
    
    def get_default_output_device(self) -> Optional[AudioDevice]:
        """获取默认输出设备"""
        try:
            info = self._pyaudio.get_default_output_device_info()
            return AudioDevice(
                index=info['index'],
                name=info['name'],
                channels=info['maxOutputChannels'],
                sample_rate=int(info['defaultSampleRate']),
                is_input=False
            )
        except:
            return None
    
    def print_devices(self):
        """打印所有设备信息"""
        print("\n🎤 输入设备:")
        for dev in self.list_input_devices():
            default = " (默认)" if self.get_default_input_device() and self.get_default_input_device().index == dev.index else ""
            print(f"  [{dev.index}] {dev.name}{default}")
        
        print("\n🔊 输出设备:")
        for dev in self.list_output_devices():
            default = " (默认)" if self.get_default_output_device() and self.get_default_output_device().index == dev.index else ""
            print(f"  [{dev.index}] {dev.name}{default}")


class FullDuplexAgent:
    """
    全双工语音对话智能体
    
    跨平台支持：Windows 和 macOS
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
        
        # 设备管理
        self._device_manager: Optional[AudioDeviceManager] = None
        self._input_device_index: Optional[int] = None
        self._output_device_index: Optional[int] = None
        
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
        
        # 帧计数
        self._frame_count = 0
        
        # 回调
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
    
    def _get_platform_audio_params(self) -> Dict[str, Any]:
        """
        获取平台特定的音频参数
        
        Windows 和 macOS 可能需要不同的参数
        """
        params = {
            'input_format': pyaudio.paInt16,
            'output_format': pyaudio.paInt16,
            'input_frames_per_buffer': int(self.config.sample_rate * 0.03),  # 30ms
            'output_frames_per_buffer': 1024,
        }
        
        if IS_WINDOWS:
            # Windows 可能需要更大的缓冲区
            params['input_frames_per_buffer'] = 1024
            params['output_frames_per_buffer'] = 2048
        elif IS_MACOS:
            # macOS Core Audio 喜欢较小的缓冲区
            params['input_frames_per_buffer'] = 480  # 30ms at 16kHz
            params['output_frames_per_buffer'] = 512
        
        return params
    
    def _init_audio(self) -> bool:
        """初始化音频设备"""
        try:
            # 创建 PyAudio 实例
            self._pyaudio_in = pyaudio.PyAudio()
            self._pyaudio_out = pyaudio.PyAudio()
            
            # 设备管理器
            self._device_manager = AudioDeviceManager(self._pyaudio_in)
            
            # 获取默认设备
            default_in = self._device_manager.get_default_input_device()
            default_out = self._device_manager.get_default_output_device()
            
            if default_in:
                self._input_device_index = default_in.index
                logger.info(f"输入设备: {default_in.name}")
            else:
                logger.warning("未找到输入设备")
                return False
            
            if default_out:
                self._output_device_index = default_out.index
                logger.info(f"输出设备: {default_out.name}")
            else:
                logger.warning("未找到输出设备")
                return False
            
            logger.info(f"✅ 音频设备初始化成功 ({platform.system()})")
            return True
            
        except Exception as e:
            logger.error(f"❌ 音频设备初始化失败: {e}")
            return False
    
    def select_devices(self):
        """交互式选择音频设备"""
        if not self._device_manager:
            print("请先初始化音频系统")
            return
        
        self._device_manager.print_devices()
        
        # 选择输入设备
        try:
            in_idx = input("\n选择输入设备编号 (回车使用默认): ").strip()
            if in_idx:
                self._input_device_index = int(in_idx)
        except:
            pass
        
        # 选择输出设备
        try:
            out_idx = input("选择输出设备编号 (回车使用默认): ").strip()
            if out_idx:
                self._output_device_index = int(out_idx)
        except:
            pass
    
    def _start_input_stream(self) -> bool:
        """启动麦克风输入流"""
        if self._input_stream is not None:
            return True
        
        try:
            params = self._get_platform_audio_params()
            
            self._input_stream = self._pyaudio_in.open(
                rate=self.config.sample_rate,
                channels=self.config.channels,
                format=params['input_format'],
                input=True,
                input_device_index=self._input_device_index,
                frames_per_buffer=params['input_frames_per_buffer'],
                stream_callback=None
            )
            
            logger.info(f"🎤 麦克风输入流已启动 (缓冲区: {params['input_frames_per_buffer']})")
            return True
            
        except Exception as e:
            logger.error(f"❌ 启动麦克风失败: {e}")
            
            # Windows 特殊处理：尝试不同的缓冲区大小
            if IS_WINDOWS:
                for buf_size in [512, 1024, 2048, 4096]:
                    try:
                        self._input_stream = self._pyaudio_in.open(
                            rate=self.config.sample_rate,
                            channels=self.config.channels,
                            format=pyaudio.paInt16,
                            input=True,
                            input_device_index=self._input_device_index,
                            frames_per_buffer=buf_size
                        )
                        logger.info(f"🎤 麦克风输入流已启动 (缓冲区: {buf_size})")
                        return True
                    except:
                        continue
            
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
                if self._input_stream is None:
                    if not self._start_input_stream():
                        time.sleep(0.5)
                        continue
                
                # 读取音频
                try:
                    params = self._get_platform_audio_params()
                    chunk_size = params['input_frames_per_buffer']
                    audio_data = self._input_stream.read(chunk_size, exception_on_overflow=False)
                except OSError as e:
                    error_msg = str(e)
                    if "Stream closed" in error_msg or "Invalid stream" in error_msg or "Unanticipated host error" in error_msg:
                        self._input_stream = None
                        time.sleep(0.1)
                        continue
                    raise
                
                self._frame_count += 1
                
                # 根据当前状态处理
                tts_playing = self._tts_playing.is_set()
                if tts_playing:
                    self._handle_interrupt_detection(audio_data)
                else:
                    self._handle_normal_vad(audio_data)
            
            except Exception as e:
                logger.error(f"音频输入错误: {e}")
                time.sleep(0.1)
        
        logger.info("🎧 音频输入线程结束")
    
    def _handle_normal_vad(self, audio: bytes):
        """正常 VAD 处理"""
        import numpy as np
        
        # 计算当前能量
        samples = np.frombuffer(audio, dtype=np.int16)
        energy = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
        
        result = self.vad.process(audio)
        
        # 调试日志
        if result.get('is_speech') or result.get('speech_start') or result.get('speech_end'):
            logger.info(f"[VAD] energy={energy:.1f}, prob={result.get('speech_prob', 0):.3f}, "
                       f"is_speech={result.get('is_speech')}, speech_end={result.get('speech_end')}")
        
        if result.get('is_speech'):
            self._audio_buffer.extend(audio)
            if self.asr.is_connected:
                self.asr.send(audio)
            self._set_state(AgentState.LISTENING)
        
        elif result.get('speech_end') and len(self._audio_buffer) > 0:
            self._set_state(AgentState.THINKING)
            time.sleep(0.3)
            
            self.asr.stop()
            self._process_user_input()
            self.asr.start()
            
            self._audio_buffer = bytearray()
            self.vad.normal_vad.reset()
    
    def _handle_interrupt_detection(self, audio: bytes):
        """打断检测处理"""
        import numpy as np
        
        # 计算能量
        samples = np.frombuffer(audio, dtype=np.int16)
        energy = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
        
        result = self.vad.process_for_interrupt(audio)
        
        # 每帧都输出调试日志
        logger.info(f"[打断检测] frame={self._frame_count}, energy={energy:.1f}, "
                   f"prob={result.get('speech_prob', 0):.3f}, "
                   f"is_speech={result.get('is_speech')}, "
                   f"speech_start={result.get('speech_start')}")
        
        if result.get('is_speech'):
            self._interrupt_audio.extend(audio)
            if self.asr.is_connected:
                self.asr.send(audio)
        
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
        audio = self.tts.synthesize(text)
        if not audio:
            self._set_state(AgentState.IDLE)
            return
        
        if len(audio) > 44 and audio[:4] == b'RIFF':
            audio = audio[44:]
        
        logger.info(f"🔊 播放 TTS ({len(audio)} 字节)")
        
        self._set_state(AgentState.SPEAKING)
        self._tts_playing.set()
        self._stop_playback.clear()
        self._should_interrupt.clear()
        self._interrupt_audio = bytearray()
        self.vad.set_tts_playing(True)
        self.vad.barge_in_detector.set_tts_state(True)
        
        interrupted = False
        
        try:
            params = self._get_platform_audio_params()
            
            stream = self._pyaudio_out.open(
                rate=self.config.tts_sample_rate,
                channels=1,
                format=params['output_format'],
                output=True,
                output_device_index=self._output_device_index,
                frames_per_buffer=params['output_frames_per_buffer']
            )
            
            chunk_size = params['output_frames_per_buffer']
            for i in range(0, len(audio), chunk_size):
                if self._stop_playback.is_set():
                    logger.info("⚡ TTS 被打断")
                    interrupted = True
                    break
                
                chunk = audio[i:i + chunk_size]
                stream.write(chunk, exception_on_underflow=False)
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            logger.error(f"播放错误: {e}")
        
        finally:
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
        
        time.sleep(0.5)
        self.asr.stop()
        time.sleep(0.2)
        
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
    
    async def run(self, interactive: bool = False):
        """
        运行智能体
        
        Args:
            interactive: 是否交互式选择设备
        """
        print("=" * 60)
        print(f"🎙️  全双工语音对话智能体 v2.2")
        print(f"🖥️  平台: {platform.system()} {platform.release()}")
        print("=" * 60)
        print(f"📦 模型: {self.config.llm_model}")
        print(f"🎵 音色: {self.config.tts_voice}")
        print(f"🎤 VAD: {self.vad.vad_name}")
        print(f"⚡ 打断检测: {'启用' if self.config.barge_in_enabled else '禁用'}")
        print("-" * 60)
        print("💡 提示:")
        print("   - 说话后停顿 0.5 秒自动提交")
        print("   - TTS 播放时可说话打断")
        print("-" * 60)
        print("🛑 Ctrl+C 退出")
        print("=" * 60)
        
        # 初始化音频
        if not self._init_audio():
            print("❌ 音频设备初始化失败")
            return
        
        # 交互式选择设备
        if interactive:
            self.select_devices()
        
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
            try:
                self._pyaudio_in.terminate()
            except:
                pass
        if self._pyaudio_out:
            try:
                self._pyaudio_out.terminate()
            except:
                pass
        
        logger.info("资源清理完成")
    
    def stop(self):
        """停止智能体"""
        self._is_running = False
        self._stop_playback.set()
    
    @property
    def is_running(self) -> bool:
        return self._is_running