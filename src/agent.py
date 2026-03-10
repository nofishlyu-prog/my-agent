"""
全双工语音智能体模块

支持真正的全双工模式：
- 边说边听：TTS 播放时持续检测用户语音
- 即时打断：检测到用户说话时立即停止 TTS
- 流式处理：ASR 实时识别打断语音
"""

import asyncio
import threading
import time
import logging
from queue import Queue, Empty
from typing import Optional

try:
    import pyaudio
except ImportError as e:
    raise ImportError("请安装 pyaudio: pip install pyaudio") from e

from .config import Config
from .state import AgentState
from .vad import VoiceActivityDetector
from .asr import SpeechRecognizer
from .llm import LanguageModel
from .tts import TextToSpeech
from .interrupt import SemanticInterruptDetector

logger = logging.getLogger(__name__)


class FullDuplexAgent:
    """全双工语音对话智能体"""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.vad = VoiceActivityDetector(self.config)
        self.asr = SpeechRecognizer(self.config)
        self.llm = LanguageModel(self.config)
        self.tts = TextToSpeech(self.config)
        self.interrupt_detector = SemanticInterruptDetector(self.config)

        # 使用两个独立的 PyAudio 实例，避免输入输出冲突
        self.audio_input = pyaudio.PyAudio()
        self.audio_output = pyaudio.PyAudio()

        self.state = AgentState.IDLE
        self.is_running = False

        self.audio_buffer = bytearray()

        # 打断控制
        self.should_interrupt = threading.Event()
        self.is_tts_playing = threading.Event()
        self.interrupt_audio = bytearray()  # 打断时的音频缓冲

        # 音频队列
        self._audio_queue = Queue(maxsize=100)
        self._input_stream = None
        self._output_stream = None

        # TTS 播放控制
        self._tts_thread: Optional[threading.Thread] = None
        self._stop_tts = threading.Event()

        # ASR 管理
        self._asr_lock = threading.Lock()
        self._asr_enabled = True

    def _set_state(self, new_state: AgentState):
        """更新状态"""
        if self.state != new_state:
            logger.info(f"状态：{self.state.value} → {new_state.value}")
            self.state = new_state

    def _start_input_stream(self):
        """启动麦克风输入流"""
        if self._input_stream is None or not self._input_stream.is_active():
            self._input_stream = self.audio_input.open(
                rate=self.config.sample_rate,
                channels=self.config.channels,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.config.chunk_size
            )
            logger.info("🎤 麦克风输入流已启动")

    def _stop_input_stream(self):
        """停止麦克风输入流"""
        if self._input_stream is not None:
            try:
                self._input_stream.stop_stream()
                self._input_stream.close()
            except:
                pass
            self._input_stream = None
            logger.info("🎤 麦克风输入流已停止")

    def _audio_input_thread(self):
        """音频输入线程 - 核心全双工处理"""
        logger.info("🎤 音频输入线程启动")
        frame_count = 0

        while self.is_running:
            try:
                # 如果输入流未启动，等待
                if self._input_stream is None:
                    time.sleep(0.05)
                    continue

                # 读取音频数据
                in_data = self._input_stream.read(
                    self.config.chunk_size,
                    exception_on_overflow=False
                )
                frame_count += 1

                # ====== TTS 播放中 - 打断检测模式 ======
                if self.is_tts_playing.is_set() and self.config.barge_in_enabled:
                    vad_result = self.vad.process_for_interrupt(in_data)

                    # 调试日志（每 30 帧或触发时）
                    if frame_count <= 30 or frame_count % 30 == 0 or vad_result.get('speech_start'):
                        logger.info(
                            f"[打断检测] frame={frame_count}, phase={vad_result.get('phase')}, "
                            f"energy={vad_result['energy']:.1f}, "
                            f"baseline={vad_result.get('baseline', 0):.1f}, "
                            f"ratio={vad_result.get('energy_ratio', 1.0):.2f}, "
                            f"threshold={vad_result.get('dynamic_threshold', 0):.1f}, "
                            f"is_speech={vad_result['is_speech']}, "
                            f"high_frames={vad_result.get('interrupt_frames', 0)}"
                        )

                    # 检测到用户语音，发送给 ASR
                    if vad_result['is_speech']:
                        with self._asr_lock:
                            if self._asr_enabled and self.asr.is_connected:
                                self.asr.send(in_data)
                                # 缓存打断时的音频
                                self.interrupt_audio.extend(in_data)

                    # 检测到打断触发
                    if vad_result['speech_start']:
                        self.should_interrupt.set()
                        logger.info(f"⚡ 触发打断！energy={vad_result['energy']:.1f}")

                # ====== 正常模式 ======
                else:
                    vad_result = self.vad.process(in_data)

                    if vad_result.get('ignored', False):
                        # TTS 刚停止，抑制期
                        pass
                    elif vad_result['is_speech']:
                        self.audio_buffer.extend(in_data)
                        with self._asr_lock:
                            if self._asr_enabled and self.asr.is_connected:
                                self.asr.send(in_data)
                        self._set_state(AgentState.LISTENING)

                    elif vad_result['speech_end']:
                        if len(self.audio_buffer) > 0:
                            # 等待 ASR 完成识别
                            time.sleep(0.5)
                            self.asr.stop()
                            time.sleep(0.3)
                            self._set_state(AgentState.THINKING)
                            self._process_dialog()
                            self.asr.start()
                        self.audio_buffer = bytearray()
                        self.vad.reset()

            except OSError as e:
                # 输入流被停止时会出现这个错误，忽略
                if "Stream closed" in str(e) or "Invalid stream" in str(e):
                    time.sleep(0.05)
                    continue
                logger.error(f"读取音频错误: {e}")
            except Exception as e:
                logger.error(f"音频输入错误: {e}")
                time.sleep(0.05)

        logger.info("🎤 音频输入线程结束")

    def _process_dialog(self):
        """处理对话"""
        user_input = self.asr.get_result(timeout=0.5)

        if not user_input or not user_input.strip():
            logger.warning("没有识别到有效语音")
            self._set_state(AgentState.IDLE)
            return

        logger.info(f"📝 识别文本：{user_input}")
        print(f"\n👤 你：{user_input}")

        self._set_state(AgentState.THINKING)
        response = self.llm.chat(user_input)

        print(f"🤖 助手：{response}")

        if response:
            self._play_response(response)

        self.asr.clear()

    def _play_response(self, text: str):
        """播放 TTS 响应 - 支持即时打断"""
        audio = self.tts.synthesize(text)
        if not audio:
            self._set_state(AgentState.IDLE)
            return

        # 跳过 WAV 头
        if len(audio) > 44 and audio[:4] == b'RIFF':
            audio = audio[44:]

        logger.info(f"[TTS] 音频 {len(audio)} 字节，开始播放")

        # 设置播放状态
        self.is_tts_playing.set()
        self.vad.set_tts_playing(True)
        self._set_state(AgentState.SPEAKING)
        self.should_interrupt.clear()
        self._stop_tts.clear()
        self.interrupt_audio = bytearray()

        interrupted = False
        stream = None

        try:
            # 使用独立的输出 PyAudio 实例
            stream = self.audio_output.open(
                rate=self.config.tts_sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                output=True,
                frames_per_buffer=512
            )

            chunk_size = 512
            total_chunks = len(audio) // chunk_size + 1
            chunks_played = 0

            for i in range(0, len(audio), chunk_size):
                # 检查打断 - 更频繁的检查点
                if self.should_interrupt.is_set():
                    logger.info("⚡ TTS 被打断！")
                    interrupted = True
                    break

                chunk = audio[i:i + chunk_size]
                stream.write(chunk, exception_on_underflow=False)
                chunks_played += 1

                # 每播放 5 个 chunk 检查一次打断
                if chunks_played % 5 == 0 and self.should_interrupt.is_set():
                    interrupted = True
                    break

            logger.info(f"[TTS] 播放结束: {chunks_played}/{total_chunks}, interrupted={interrupted}")

        except Exception as e:
            logger.error(f"播放错误：{e}")
        finally:
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except:
                    pass

            # 清除状态
            self.is_tts_playing.clear()
            self.vad.set_tts_playing(False)

            if interrupted:
                self._handle_interrupt()
            else:
                self._set_state(AgentState.IDLE)

    def _handle_interrupt(self):
        """处理打断 - 立即响应用户的新输入"""
        logger.info("[打断] 处理中...")
        self.should_interrupt.clear()

        # 等待 ASR 完成识别
        time.sleep(0.5)

        # 停止 ASR 并获取结果
        self.asr.stop()
        time.sleep(0.2)

        user_input = self.asr.get_result(timeout=0.3)

        # 如果没有识别结果，尝试使用缓存的音频重新识别
        if not user_input or not user_input.strip():
            # 使用打断时缓存的音频
            if len(self.interrupt_audio) > 0:
                logger.info(f"[打断] 尝试重新识别缓存音频: {len(self.interrupt_audio)} 字节")
                # 这里可以尝试重新发送音频到 ASR，但实时 ASR 通常不支持
                # 所以我们依赖 ASR 的临时结果
                with self.asr._lock:
                    user_input = self.asr._partial_text or self.asr._current_text
                    self.asr._current_text = ""
                    self.asr._partial_text = ""

        # 清空音频缓存
        self.interrupt_audio = bytearray()

        if user_input and user_input.strip():
            logger.info(f"📝 打断识别：{user_input}")
            print(f"\n⚡ 打断：{user_input}")

            self.asr.start()
            self._set_state(AgentState.THINKING)
            response = self.llm.chat(user_input)
            print(f"🤖 助手：{response}")

            if response:
                self._play_response(response)
            else:
                self._set_state(AgentState.IDLE)
        else:
            logger.info("[打断] 未识别到有效语音")
            self.asr.start()
            self._set_state(AgentState.IDLE)

    async def run(self):
        """运行智能体"""
        print("=" * 70)
        print("🎙️  全双工语音对话智能体")
        print("=" * 70)
        print(f"📦 模型：{self.config.llm_model}")
        print(f"🔄 架构：VAD → ASR → LLM → TTS")
        print(f"🎵 TTS 音色：{self.config.tts_voice}")
        print(f"🎤 VAD 阈值：{self.config.vad_threshold}")
        print(f"⚡ 打断模式：{'启用' if self.config.barge_in_enabled else '禁用'}")
        print("-" * 70)
        print("💡 使用说明：")
        print("   - 说话后停顿 0.5 秒自动提交")
        print("   - TTS 播放时可直接说话打断")
        print("   - 打断时需要说话声音足够大（超过背景噪声 30%）")
        print("-" * 70)
        print("🛑 Ctrl+C 退出")
        print("=" * 70)

        self.is_running = True
        self._set_state(AgentState.IDLE)
        self.asr.start()

        # 启动输入流
        self._start_input_stream()

        # 启动音频输入线程
        input_thread = threading.Thread(target=self._audio_input_thread, daemon=True)
        input_thread.start()

        try:
            while self.is_running:
                await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
        finally:
            self.is_running = False
            self._stop_input_stream()
            self.asr.stop()
            self.audio_input.terminate()
            self.audio_output.terminate()