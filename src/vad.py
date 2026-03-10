"""
语音活动检测 (VAD) 模块

支持多种 VAD 策略：
1. 能量检测 (简单快速)
2. WebRTC VAD (传统方法)
3. Silero VAD (深度学习，推荐)

全双工打断检测策略：
- 回声感知：检测 TTS 播放状态，动态调整阈值
- 能量突变检测：检测相对于背景的突然变化
- 持续确认：避免瞬时噪声误触发
"""

import struct
import threading
import time
import logging
from collections import deque
from typing import Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np

logger = logging.getLogger(__name__)


class VADInterface(ABC):
    """VAD 接口"""
    
    @abstractmethod
    def process(self, audio: bytes) -> Dict[str, Any]:
        """处理音频帧，返回检测结果"""
        pass
    
    @abstractmethod
    def reset(self):
        """重置状态"""
        pass


class EnergyVAD(VADInterface):
    """
    能量检测 VAD
    
    简单但有效的方法，适合资源受限环境
    """
    
    def __init__(self, config: "Config"):
        self.config = config
        self.sample_rate = config.sample_rate
        self.frame_duration_ms = 30  # 每帧 30ms
        self.frame_size = int(self.sample_rate * self.frame_duration_ms / 1000)
        
        # 状态
        self.is_speaking = False
        self.speech_frames = 0
        self.silence_frames = 0
        self.energy_history = deque(maxlen=100)
        
        # 阈值
        self.energy_threshold = config.vad_threshold
        self.min_speech_frames = int(config.vad_speech_ms / self.frame_duration_ms)
        self.min_silence_frames = int(config.vad_silence_ms / self.frame_duration_ms)
        
        # 自适应阈值
        self.adaptive_threshold = self.energy_threshold
        self.noise_floor = 0
        self._lock = threading.Lock()
    
    def _calc_energy(self, audio: bytes) -> float:
        """计算 RMS 能量"""
        samples = np.frombuffer(audio, dtype=np.int16)
        if len(samples) == 0:
            return 0.0
        return np.sqrt(np.mean(samples.astype(np.float32) ** 2))
    
    def process(self, audio: bytes) -> Dict[str, Any]:
        """处理音频帧"""
        energy = self._calc_energy(audio)
        
        with self._lock:
            self.energy_history.append(energy)
            
            # 更新噪声底（使用较低的百分位）
            if len(self.energy_history) >= 30:
                sorted_energy = sorted(self.energy_history)
                self.noise_floor = sorted_energy[int(len(sorted_energy) * 0.1)]
            
            # 自适应阈值
            self.adaptive_threshold = max(
                self.energy_threshold,
                self.noise_floor * 2
            )
        
        is_speech = energy > self.adaptive_threshold
        
        result = {
            'is_speech': is_speech,
            'speech_start': False,
            'speech_end': False,
            'energy': energy,
            'threshold': self.adaptive_threshold,
            'noise_floor': self.noise_floor,
        }
        
        # 状态机
        if is_speech:
            self.speech_frames += 1
            self.silence_frames = 0
            
            if not self.is_speaking and self.speech_frames >= self.min_speech_frames:
                self.is_speaking = True
                result['speech_start'] = True
        else:
            self.silence_frames += 1
            
            if self.is_speaking and self.silence_frames >= self.min_silence_frames:
                self.is_speaking = False
                self.speech_frames = 0
                result['speech_end'] = True
        
        return result
    
    def reset(self):
        """重置状态"""
        with self._lock:
            self.is_speaking = False
            self.speech_frames = 0
            self.silence_frames = 0
            self.energy_history.clear()


class BargeInDetector:
    """
    打断检测器
    
    核心策略：检测 TTS 播放时用户语音的能量突变
    
    关键洞察：
    - TTS 回声是"平稳"的能量，变化缓慢
    - 用户说话会产生"突发"的能量变化
    - 通过检测能量的变化率而非绝对值来识别
    """
    
    def __init__(self, config: "Config"):
        self.config = config
        
        # 帧参数
        self.frame_duration_ms = 30
        self.sample_rate = config.sample_rate
        
        # 能量历史（用于计算变化）
        self.energy_history = deque(maxlen=50)
        self.delta_history = deque(maxlen=20)  # 能量变化历史
        
        # 状态
        self._frame_count = 0
        self._baseline_energy = 0.0
        self._speaking_frames = 0
        self._is_speaking = False
        self._last_energy = 0.0
        
        # 阈值参数
        self.min_energy = getattr(config, 'barge_in_min_increment', 100)
        self.energy_ratio_threshold = getattr(config, 'barge_in_ratio_threshold', 1.5)
        self.confirm_frames = getattr(config, 'barge_in_confirm_frames', 3)
        self.delta_threshold = 50  # 能量变化阈值
        
        # TTS 状态
        self._tts_playing = False
        self._tts_start_time = 0
        self._lock = threading.Lock()
    
    def set_tts_state(self, playing: bool):
        """设置 TTS 播放状态"""
        with self._lock:
            self._tts_playing = playing
            if playing:
                self._tts_start_time = time.time()
                # 重置检测状态
                self._frame_count = 0
                self._baseline_energy = 0
                self._speaking_frames = 0
                self._is_speaking = False
                self.energy_history.clear()
                self.delta_history.clear()
    
    def _calc_energy(self, audio: bytes) -> float:
        """计算 RMS 能量"""
        samples = np.frombuffer(audio, dtype=np.int16)
        if len(samples) == 0:
            return 0.0
        return np.sqrt(np.mean(samples.astype(np.float32) ** 2))
    
    def process(self, audio: bytes) -> Dict[str, Any]:
        """
        处理音频帧，检测打断
        
        返回：
        - is_speech: 当前是否有语音
        - speech_start: 是否检测到新的语音开始（打断触发）
        - energy: 当前能量
        - baseline: 基线能量
        - reason: 触发原因
        """
        energy = self._calc_energy(audio)
        self._frame_count += 1
        
        with self._lock:
            self.energy_history.append(energy)
            current_energy = energy
        
        # 计算能量变化
        delta = abs(energy - self._last_energy)
        self.delta_history.append(delta)
        self._last_energy = energy
        
        result = {
            'is_speech': False,
            'speech_start': False,
            'energy': current_energy,
            'baseline': self._baseline_energy,
            'delta': delta,
            'frame': self._frame_count,
            'reason': None,
        }
        
        # === 阶段 1: 初始化基线（前 15 帧，约 450ms）===
        init_frames = 15
        if self._frame_count <= init_frames:
            result['reason'] = 'init'
            return result
        
        # === 阶段 2: 动态基线计算 ===
        # 使用滑动窗口的中位数作为基线（代表 TTS 回声水平）
        if len(self.energy_history) >= 10:
            sorted_energy = sorted(self.energy_history)
            self._baseline_energy = sorted_energy[len(sorted_energy) // 2]
        
        result['baseline'] = self._baseline_energy
        
        # === 阶段 3: 打断检测 ===
        # 方法 1: 能量比率检测
        if self._baseline_energy > 0:
            energy_ratio = current_energy / self._baseline_energy
        else:
            energy_ratio = 1.0
        
        # 方法 2: 能量突变检测（更可靠）
        # 用户说话时，能量会在短时间内显著增加
        avg_delta = sum(self.delta_history) / len(self.delta_history) if self.delta_history else 0
        
        # 检测条件
        above_ratio = energy_ratio > self.energy_ratio_threshold
        above_absolute = current_energy > self._baseline_energy + self.min_energy
        sudden_increase = delta > self.delta_threshold and current_energy > self._baseline_energy
        
        # 综合判断：需要满足多个条件
        is_potential_interrupt = (above_ratio and above_absolute) or sudden_increase
        
        result['energy_ratio'] = energy_ratio
        result['above_ratio'] = above_ratio
        result['above_absolute'] = above_absolute
        result['sudden_increase'] = sudden_increase
        result['avg_delta'] = avg_delta
        
        # 持续确认
        if is_potential_interrupt:
            self._speaking_frames += 1
            result['is_speech'] = True
            
            if self._speaking_frames >= self.confirm_frames:
                if not self._is_speaking:
                    self._is_speaking = True
                    result['speech_start'] = True
                    result['reason'] = 'interrupt_detected'
                    logger.info(
                        f"⚡ 打断检测: energy={current_energy:.1f}, "
                        f"baseline={self._baseline_energy:.1f}, ratio={energy_ratio:.2f}, "
                        f"delta={delta:.1f}"
                    )
        else:
            self._speaking_frames = max(0, self._speaking_frames - 1)
            if self._speaking_frames == 0:
                self._is_speaking = False
        
        result['speaking_frames'] = self._speaking_frames
        
        return result
    
    def reset(self):
        """重置状态"""
        with self._lock:
            self._frame_count = 0
            self._baseline_energy = 0
            self._speaking_frames = 0
            self._is_speaking = False
            self._last_energy = 0
            self.energy_history.clear()
            self.delta_history.clear()


class VoiceActivityDetector:
    """
    统一的语音活动检测器
    
    整合正常 VAD 和打断检测
    """
    
    def __init__(self, config: "Config"):
        self.config = config
        
        # 正常 VAD
        self.normal_vad = EnergyVAD(config)
        
        # 打断检测器
        self.barge_in_detector = BargeInDetector(config)
        
        # TTS 状态
        self._tts_playing = False
        self._lock = threading.Lock()
    
    def set_tts_playing(self, playing: bool):
        """设置 TTS 播放状态"""
        with self._lock:
            self._tts_playing = playing
            self.barge_in_detector.set_tts_state(playing)
    
    def is_tts_playing(self) -> bool:
        """检查 TTS 是否在播放"""
        with self._lock:
            return self._tts_playing
    
    def process(self, audio: bytes) -> Dict[str, Any]:
        """
        处理音频帧
        
        根据 TTS 状态自动选择检测模式
        """
        with self._lock:
            tts_playing = self._tts_playing
        
        if tts_playing and self.config.barge_in_enabled:
            # 打断检测模式
            return self.barge_in_detector.process(audio)
        else:
            # 正常 VAD 模式
            return self.normal_vad.process(audio)
    
    def process_for_interrupt(self, audio: bytes) -> Dict[str, Any]:
        """专门用于打断检测"""
        return self.barge_in_detector.process(audio)
    
    def reset(self):
        """重置所有状态"""
        self.normal_vad.reset()
        self.barge_in_detector.reset()


# 兼容旧接口
class SileroVAD:
    """
    Silero VAD 包装器（需要 torch）
    
    更准确，但需要额外依赖
    """
    
    def __init__(self, config: "Config"):
        self.config = config
        self._model = None
        self._initialized = False
    
    def _init_model(self):
        """延迟初始化模型"""
        if self._initialized:
            return
        
        try:
            import torch
            
            # 加载 Silero VAD 模型
            self._model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            self._model.eval()
            self._initialized = True
            logger.info("✅ Silero VAD 模型加载成功")
        except Exception as e:
            logger.warning(f"⚠️ Silero VAD 加载失败，回退到能量检测: {e}")
            self._initialized = True
            self._model = None
    
    def process(self, audio: bytes) -> Dict[str, Any]:
        """处理音频帧"""
        self._init_model()
        
        if self._model is None:
            # 回退到能量检测
            return EnergyVAD(self.config).process(audio)
        
        import torch
        
        # 转换音频
        samples = np.frombuffer(audio, dtype=np.int16)
        float_samples = samples.astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(float_samples)
        
        # 检测
        with torch.no_grad():
            speech_prob = self._model(audio_tensor, self.config.sample_rate).item()
        
        is_speech = speech_prob > 0.5
        
        return {
            'is_speech': is_speech,
            'speech_prob': speech_prob,
            'speech_start': False,
            'speech_end': False,
        }