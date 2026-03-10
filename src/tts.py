"""
语音合成 (TTS) 模块
"""

import logging

try:
    import dashscope
    from dashscope.audio.tts_v2 import SpeechSynthesizer, AudioFormat
except ImportError as e:
    raise ImportError("请安装 dashscope: pip install dashscope") from e

logger = logging.getLogger(__name__)


class TextToSpeech:
    """语音合成器"""

    def __init__(self, config: "Config"):
        self.config = config
        dashscope.api_key = config.api_key

    def synthesize(self, text: str) -> bytes:
        """合成语音"""
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