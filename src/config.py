"""
配置管理模块
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """系统配置"""
    api_key: str = "sk-*******************"

    # 音频参数
    sample_rate: int = 16000
    tts_sample_rate: int = 22050
    chunk_size: int = 960  # 60ms at 16kHz
    channels: int = 1

    # VAD 参数（正常模式）
    vad_threshold: int = 100  # 能量阈值
    vad_silence_ms: int = 500  # 静音检测时长
    vad_speech_ms: int = 200   # 语音确认时长

    # 打断检测参数（全双工模式）
    barge_in_enabled: bool = True
    barge_in_baseline_frames: int = 20      # 基线估计帧数
    barge_in_confirm_frames: int = 3        # 确认帧数
    barge_in_min_increment: int = 80        # 最小能量增量
    barge_in_ratio_threshold: float = 1.3   # 能量比率阈值（比基线高 30%）

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

    # 打断关键词（语义打断）
    interrupt_keywords: List[str] = field(default_factory=lambda: [
        '等等', '等一下', '停下', '停一下', '停',
        '不要说了', '别说', '别说了',
        '不对', '错了', '不是这样', '不是',
        '取消', '停止', '闭嘴', '打断', '慢点', '重新说'
    ])

    # 调试模式
    debug_mode: bool = False

    @classmethod
    def from_json(cls, path: str = "config.json") -> "Config":
        """从 JSON 文件加载配置"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            config = cls()
            for key, value in data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    logger.warning(f"未知配置项: {key}")
            return config
        except Exception as e:
            logger.warning(f"无法加载配置文件 {path}: {e}")
            return cls()

    def to_json(self, path: str = "config.json"):
        """保存配置到 JSON 文件"""
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'api_key': self.api_key,
                'sample_rate': self.sample_rate,
                'vad_threshold': self.vad_threshold,
                'barge_in_enabled': self.barge_in_enabled,
                'barge_in_baseline_frames': self.barge_in_baseline_frames,
                'barge_in_confirm_frames': self.barge_in_confirm_frames,
                'barge_in_min_increment': self.barge_in_min_increment,
                'barge_in_ratio_threshold': self.barge_in_ratio_threshold,
                'asr_model': self.asr_model,
                'llm_model': self.llm_model,
                'tts_model': self.tts_model,
                'tts_voice': self.tts_voice,
                'enable_full_duplex': self.enable_full_duplex,
                'interrupt_keywords': self.interrupt_keywords,
                'debug_mode': self.debug_mode,
            }, f, ensure_ascii=False, indent=2)