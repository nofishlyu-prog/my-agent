#!/usr/bin/env python3
"""
全双工语音对话智能体 - 优化版
架构：VAD → ASR → LLM → TTS
"""

import asyncio
import queue
import threading
import struct
import time
import re
import json
import logging
import base64
from pathlib import Path
from typing import Optional, AsyncGenerator, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

try:
    import pyaudio
    import dashscope
    from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult
    from dashscope.audio.tts_v2 import SpeechSynthesizer, AudioFormat
    from openai import OpenAI
except ImportError as e:
    print("❌ 请安装依赖：pip install -r requirements.txt")
    print(f"   缺少：{e}")
    exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ==================== 配置管理 ====================

@dataclass
class Config:
    """系统配置"""
    api_key: str = "sk-*******************"

    # 音频参数
    sample_rate: int = 16000
    tts_sample_rate: int = 22050
    chunk_size: int = 960
    channels: int = 1

    # VAD 参数 - 阈值已降低到 50
    vad_threshold: int = 50
    vad_silence_ms: int = 500
    vad_speech_ms: int = 200

    # ASR 参数
    asr_model: str = "paraformer-realtime-v2"

    # LLM 参数
    llm_model: str = "qwen3-omni-flash-2025-12-01"
    llm_max_history: int = 10
    llm_system_prompt: str = (
        "你是友好的语音助手，名字叫小智。"
        "回答要简洁、自然、口语化，控制在 50 字以内。"
    )

    # TTS 参数
    tts_model: str = "cosyvoice-v1"
    tts_voice: str = "longxiaochun"

    # 全双工参数
    enable_full_duplex: bool = True
    barge_in_enabled: bool = True

    # 打断关键词
    interrupt_keywords: List[str] = field(default_factory=lambda: [
        '等等', '等一下', '停下', '停一下', '停',
        '不要说了', '别说', '别说了',
        '不对', '错了', '不是这样', '不是',
        '取消', '停止', '闭嘴', '打断', '慢点', '重新说'
    ])

    @classmethod
    def from_json(cls, path: str = "config.json") -> "Config":
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            config = cls()
            for key, value in data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            return config
        except Exception as e:
            logger.warning(f"无法加载配置文件 {path}: {e}")
            return cls()


# ==================== 状态机 ====================

class AgentState(Enum):
    IDLE = "空闲"
    LISTENING = "聆听"
    THINKING = "思考"
    SPEAKING = "说话"
    INTERRUPTED = "被打断"


# ==================== VAD 模块 ====================

class VoiceActivityDetector:
    def __init__(self, config: Config):
        self.config = config
        self.is_speaking = False
        self.silence_start: Optional[float] = None
        self.energy_history = deque(maxlen=50)
        self.min_speech_frames = int(config.vad_speech_ms / 60)
        self.silence_frames = int(config.vad_silence_ms / 60)
        self.speech_frames = 0
        self.silence_count = 0
        self.tts_playing = False
        self.suppress_count = 0
        self._lock = threading.Lock()

    def _calc_energy(self, audio: bytes) -> float:
        samples = struct.unpack(f'<{len(audio)//2}h', audio)
        if not samples:
            return 0.0
        return (sum(s * s for s in samples) / len(samples)) ** 0.5

    def set_tts_playing(self, playing: bool):
        with self._lock:
            if not playing and self.tts_playing:
                self.suppress_count = 3
            self.tts_playing = playing

    def should_ignore(self) -> bool:
        with self._lock:
            if self.tts_playing:
                return True
            if self.suppress_count > 0:
                self.suppress_count -= 1
                return True
            return False

    def process(self, audio: bytes) -> Dict[str, Any]:
        if self.should_ignore():
            return {'is_speech': False, 'speech_start': False, 'speech_end': False, 'energy': 0, 'ignored': True}

        energy = self._calc_energy(audio)
        with self._lock:
            self.energy_history.append(energy)

        threshold = self.config.vad_threshold
        is_speech = energy > threshold

        result = {'is_speech': is_speech, 'speech_start': False, 'speech_end': False, 'energy': energy, 'ignored': False}

        if is_speech and not self.is_speaking:
            self.speech_frames += 1
            if self.speech_frames >= self.min_speech_frames:
                self.is_speaking = True
                self.silence_start = None
                self.silence_count = 0
                result['speech_start'] = True
        elif is_speech and self.is_speaking:
            self.silence_start = None
            self.silence_count = 0
        elif not is_speech and self.is_speaking:
            self.silence_count += 1
            if self.silence_start is None:
                self.silence_start = time.time()
            if self.silence_count >= self.silence_frames:
                self.is_speaking = False
                self.speech_frames = 0
                self.silence_count = 0
                result['speech_end'] = True
        elif not is_speech and not self.is_speaking:
            self.speech_frames = 0

        return result

    def reset(self):
        self.is_speaking = False
        self.silence_start = None
        self.speech_frames = 0
        self.silence_count = 0
        with self._lock:
            self.tts_playing = False
            self.suppress_count = 0


# ==================== ASR 模块 ====================

class SpeechRecognizer:
    def __init__(self, config: Config):
        self.config = config
        dashscope.api_key = config.api_key

        self.result_queue = queue.Queue()
        self._current_text = ""
        self.recognizer = None
        self.is_connected = False
        self._lock = threading.Lock()
        self._callback_ref = None
        self._partial_text = ""
        self._audio_buffer = bytearray()  # 本地缓冲

    def _create_callback(self):
        recognizer = self

        class Callback(RecognitionCallback):
            def on_open(self):
                recognizer.is_connected = True
                logger.info("🎤 ASR 已连接")

            def on_close(self):
                recognizer.is_connected = False

            def on_event(self, result: RecognitionResult):
                sentence = result.get_sentence()
                if sentence:
                    text = sentence.get('text', '')
                    is_final = sentence.get('is_final', False)
                    if text:
                        with recognizer._lock:
                            recognizer._current_text = text
                            recognizer._partial_text = text
                        status = "最终" if is_final else "临时"
                        print(f"\r👂 [{status}] {text}   ", end='', flush=True)

            def on_complete(self):
                with recognizer._lock:
                    final_text = recognizer._partial_text
                    if final_text:
                        recognizer.result_queue.put(final_text)
                        logger.info(f"📝 ASR 识别：{final_text}")
                    recognizer._partial_text = ""
                    recognizer._current_text = ""

            def on_error(self, result):
                logger.error(f"ASR 错误：{result}")
                recognizer.is_connected = False

        self._callback_ref = Callback()
        return self._callback_ref

    def start(self):
        if self.recognizer:
            try:
                self.recognizer.stop()
            except:
                pass

        self.recognizer = Recognition(
            model=self.config.asr_model,
            format='pcm',
            sample_rate=self.config.sample_rate,
            callback=self._create_callback()
        )
        self.recognizer.start()

    def send(self, audio: bytes):
        if self.is_connected and self.recognizer:
            self.recognizer.send_audio_frame(audio)

    def stop(self):
        if self.recognizer:
            self.recognizer.stop()
            self.is_connected = False

    def restart(self):
        """重启 ASR 连接"""
        self.stop()
        self.clear()
        time.sleep(0.1)
        self.start()

    def get_result(self, timeout: float = 0.5) -> Optional[str]:
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            with self._lock:
                text = self._current_text
                self._current_text = ""
            return text if text else None

    def clear(self):
        """清空当前识别状态"""
        with self._lock:
            self._current_text = ""
            self._partial_text = ""
        # 清空队列
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except:
                break


# ==================== LLM 模块 ====================

class LanguageModel:
    def __init__(self, config: Config):
        self.config = config
        self.conversation_history: List[Dict[str, str]] = []
        self._lock = threading.Lock()

        self.client = OpenAI(
            api_key=config.api_key,
            base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
        )

    async def chat(self, user_input: str) -> str:
        with self._lock:
            self.conversation_history.append({"role": "user", "content": user_input})
            if len(self.conversation_history) > self.config.llm_max_history * 2:
                self.conversation_history = self.conversation_history[-self.config.llm_max_history * 2:]

            messages = [{"role": "system", "content": self.config.llm_system_prompt}] + self.conversation_history

        try:
            response = self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=messages
            )

            if response.choices and len(response.choices) > 0:
                result = response.choices[0].message.content

                with self._lock:
                    self.conversation_history.append({"role": "assistant", "content": result})

                return result
            return ""
        except Exception as e:
            logger.error(f"LLM 错误：{e}")
            return "抱歉，我遇到了一些问题。"

    def clear_history(self):
        with self._lock:
            self.conversation_history.clear()


# ==================== TTS 模块 ====================

class TextToSpeech:
    def __init__(self, config: Config):
        self.config = config
        dashscope.api_key = config.api_key

    def synthesize(self, text: str) -> bytes:
        if not text or not text.strip():
            return b""
        try:
            synthesizer = SpeechSynthesizer(
                model=self.config.tts_model,
                voice=self.config.tts_voice,
                format=AudioFormat.WAV_22050HZ_MONO_16BIT
            )
            return synthesizer.call(text)
        except Exception as e:
            logger.error(f"TTS 错误：{e}")
            return b""


# ==================== 语义打断检测 ====================

class SemanticInterruptDetector:
    def __init__(self, config: Config):
        self.keywords = config.interrupt_keywords
        self.interrupt_pattern = re.compile('|'.join(re.escape(k) for k in self.keywords))

    def check(self, text: str) -> Dict[str, Any]:
        if not text or not text.strip():
            return {'interrupt': False, 'keyword': None}
        match = self.interrupt_pattern.search(text)
        if match:
            return {'interrupt': True, 'keyword': match.group()}
        return {'interrupt': False, 'keyword': None}


# ==================== 全双工智能体 ====================

class FullDuplexAgent:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.vad = VoiceActivityDetector(self.config)
        self.asr = SpeechRecognizer(self.config)
        self.llm = LanguageModel(self.config)
        self.tts = TextToSpeech(self.config)
        self.interrupt_detector = SemanticInterruptDetector(self.config)

        self.audio = pyaudio.PyAudio()
        self.state = AgentState.IDLE
        self.is_running = False

        self.audio_buffer = bytearray()
        self.tts_audio_queue = queue.Queue()

        self.should_interrupt = threading.Event()
        self.is_tts_playing = False

    def _set_state(self, new_state: AgentState):
        if self.state != new_state:
            logger.info(f"状态：{self.state.value} → {new_state.value}")
            self.state = new_state

    def _audio_input_loop(self):
        stream = None
        try:
            stream = self.audio.open(
                rate=self.config.sample_rate,
                channels=self.config.channels,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.config.chunk_size
            )

            logger.info("🎤 麦克风已开启，请说话...")

            while self.is_running:
                try:
                    audio_data = stream.read(self.config.chunk_size, exception_on_overflow=False)

                    # TTS 播放中，检测打断
                    if self.is_tts_playing and self.config.barge_in_enabled:
                        vad_result = self.vad.process(audio_data)
                        # 检测到语音，触发打断
                        if vad_result['is_speech']:
                            # 发送音频给 ASR 进行识别
                            if self.asr.is_connected:
                                self.asr.send(audio_data)
                            self._set_state(AgentState.LISTENING)
                        if vad_result['speech_start']:
                            self.should_interrupt.set()
                            logger.info("⚡ 检测到打断")
                        continue

                    if not self.is_tts_playing:
                        vad_result = self.vad.process(audio_data)

                        if vad_result.get('ignored', False):
                            continue

                        if vad_result['is_speech']:
                            self.audio_buffer.extend(audio_data)
                            if self.asr.is_connected:
                                self.asr.send(audio_data)
                            self._set_state(AgentState.LISTENING)

                        if vad_result['speech_end']:
                            if len(self.audio_buffer) > 0:
                                # 等待 ASR 处理音频
                                time.sleep(0.5)

                                # 停止 ASR 接收，触发 on_complete
                                self.asr.stop()

                                # 等待 on_complete 回调
                                time.sleep(0.3)

                                self._set_state(AgentState.THINKING)
                                self._process_dialog()

                                # 重启 ASR
                                self.asr.start()

                            # 清空缓冲和状态
                            self.audio_buffer = bytearray()
                            self.vad.reset()

                except Exception as e:
                    if self.is_running:
                        logger.debug(f"输入错误：{e}")
                        # ASR 断开时自动重连
                        if not self.asr.is_connected:
                            self.asr.restart()
                        time.sleep(0.1)
        finally:
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except:
                    pass

    def _process_dialog(self):
        # 从队列获取结果（等待 on_complete）
        user_input = self.asr.get_result(timeout=0.5)

        if not user_input or not user_input.strip():
            logger.warning("没有识别到有效语音")
            self._set_state(AgentState.IDLE)
            return

        logger.info(f"📝 识别文本：{user_input}")
        print(f"\n👤 你：{user_input}")

        # 调用 LLM
        self._set_state(AgentState.THINKING)
        response = asyncio.new_event_loop().run_until_complete(self.llm.chat(user_input))

        print(f"🤖 助手：{response}")

        # 播放 TTS
        if response:
            self._play_response(response)

        # 清空 ASR 状态，防止累积
        self.asr.clear()

    def _play_response(self, text: str):
        audio = self.tts.synthesize(text)
        if not audio:
            self._set_state(AgentState.IDLE)
            return

        self.is_tts_playing = True
        self.vad.set_tts_playing(True)
        self._set_state(AgentState.SPEAKING)

        interrupted = False
        try:
            stream = self.audio.open(
                rate=self.config.tts_sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                output=True
            )

            chunk_size = 4096
            for i in range(0, len(audio), chunk_size):
                if self.should_interrupt.is_set():
                    logger.info("⚡ 被打断")
                    interrupted = True
                    break
                stream.write(audio[i:i+chunk_size])

            stream.stop_stream()
            stream.close()

            # 如果被打断，处理用户的语音
            if interrupted:
                # 等待 ASR 识别完成
                time.sleep(0.5)
                self.asr.stop()
                time.sleep(0.3)

                # 获取识别结果并处理
                user_input = self.asr.get_result(timeout=0.3)
                if not user_input:
                    with self.asr._lock:
                        user_input = self.asr._current_text
                        self.asr._current_text = ""

                if user_input and user_input.strip():
                    logger.info(f"📝 打断识别：{user_input}")
                    print(f"\n👤 你：{user_input}")
                    # 调用 LLM 响应
                    response = asyncio.new_event_loop().run_until_complete(self.llm.chat(user_input))
                    print(f"🤖 助手：{response}")
                    # 播放新响应
                    self._play_response(response)
                    return

        except Exception as e:
            logger.error(f"播放错误：{e}")
        finally:
            self.is_tts_playing = False
            self.vad.set_tts_playing(False)
            self.should_interrupt.clear()

        self._set_state(AgentState.IDLE)

    async def run(self):
        print("=" * 70)
        print("🎙️  全双工语音对话智能体")
        print("=" * 70)
        print(f"📦 模型：{self.config.llm_model}")
        print(f"🔄 架构：VAD → ASR → LLM → TTS")
        print(f"🎵 TTS 音色：{self.config.tts_voice}")
        print(f"🎤 VAD 阈值：{self.config.vad_threshold}")
        print("-" * 70)
        print("⚡ 说话后停顿 0.5 秒自动提交")
        print("⚡ 播放时说打断关键词可抢话")
        print("-" * 70)
        print("🛑 Ctrl+C 退出")
        print("=" * 70)

        self.is_running = True
        self._set_state(AgentState.IDLE)
        self.asr.start()

        input_thread = threading.Thread(target=self._audio_input_loop, daemon=True)
        input_thread.start()

        try:
            while self.is_running:
                await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
        finally:
            self.is_running = False
            self.asr.stop()
            self.audio.terminate()


async def main():
    config = Config.from_json("config.json")

    print("\n" + "=" * 70)
    print("🎙️  全双工语音对话智能体")
    print("=" * 70)
    print(f"\n当前模型：{config.llm_model}")
    print(f"VAD 阈值：{config.vad_threshold}")
    print("\n1. 测试模式 (文字)")
    print("2. 语音对话\n")

    choice = input("选择：").strip()

    if choice == '1':
        await TestMode(config).run()
    else:
        await FullDuplexAgent(config).run()


class TestMode:
    def __init__(self, config: Config):
        self.config = config
        self.llm = LanguageModel(config)
        self.tts = TextToSpeech(config)
        self.interrupt_detector = SemanticInterruptDetector(config)

    def _play_audio(self, text: str):
        audio = self.tts.synthesize(text)
        if not audio:
            return
        try:
            p = pyaudio.PyAudio()
            stream = p.open(rate=self.config.tts_sample_rate, channels=1, format=pyaudio.paInt16, output=True)
            stream.write(audio[44:])
            stream.stop_stream()
            stream.close()
            p.terminate()
        except Exception as e:
            logger.error(f"播放错误：{e}")

    async def run(self):
        print("=" * 70)
        print("🧪 测试模式 - Qwen3-Omni")
        print("=" * 70)

        print("\n📊 语义打断测试：")
        test_cases = [
            ("等等，我想问一下", True),
            ("嗯，那个...", False),
            ("停下，不对", True),
            ("今天天气不错", False),
            ("闭嘴", True),
            ("重新说一遍", True),
        ]

        passed = 0
        for text, expected in test_cases:
            result = self.interrupt_detector.check(text)
            is_correct = result['interrupt'] == expected
            if is_correct:
                passed += 1
            status = "✅" if is_correct else "❌"
            action = "⚡ 打断" if result['interrupt'] else "✓ 不打断"
            print(f"  {status} {action}: \"{text}\"")

        print(f"\n测试通过：{passed}/{len(test_cases)}")

        self._play_audio("你好，我是你的语音助手小智")
        print("\n💬 文字对话模式，输入 quit 退出\n")

        while True:
            try:
                text = input("你：").strip()
                if text.lower() == 'quit':
                    break
                if not text:
                    continue

                result = self.interrupt_detector.check(text)
                if result['interrupt']:
                    print(f"⚡ {result['reason']}")
                    self._play_audio("好的，请问有什么事？")
                    continue

                print("助手：", end='', flush=True)
                response = await self.llm.chat(text)
                print(response)
                self._play_audio(response)
            except KeyboardInterrupt:
                break
        print("\n👋 再见！")


if __name__ == '__main__':
    asyncio.run(main())
