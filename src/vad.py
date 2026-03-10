"""
语音活动检测 (VAD) 模块

支持多种 VAD 策略：
1. Silero VAD (深度学习，默认，高准确率)
2. 能量检测 (简单快速，备用)
3. WebRTC VAD (传统方法)

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
from enum import Enum

try:
    import numpy as np
except ImportError:
    raise ImportError("请安装 numpy: pip install numpy")

logger = logging.getLogger(__name__)


class VADType(Enum):
    """VAD 类型"""
    SILERO = "silero"
    ENERGY = "energy"


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


class SileroVAD(VADInterface):
    """
    Silero VAD - 深度学习语音活动检测
    
    特点：
    - 高准确率（>95%）
    - 低延迟（~1ms）
    - 自动处理噪声
    - 支持流式处理
    
    需要：pip install torch
    """
    
    def __init__(self, config: "Config"):
        self.config = config
        self.sample_rate = config.sample_rate
        
        # 模型和状态
        self._model = None
        self._utils = None
        self._h = None  # 隐藏状态
        self._c = None  # Cell 状态
        self._initialized = False
        self._init_lock = threading.Lock()
        
        # 检测状态
        self.is_speaking = False
        self._speech_prob_history = deque(maxlen=10)
        
        # 帧参数
        self.frame_size = 512  # Silero 推荐 512, 1024, 1536
        
    def _init_model(self):
        """初始化 Silero 模型"""
        with self._init_lock:
            if self._initialized:
                return
            
            try:
                import torch
                
                logger.info("正在加载 Silero VAD 模型...")
                
                # 加载模型 - 新版 API
                model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    trust_repo=True
                )
                
                self._model = model
                self._utils = utils
                self._model.eval()
                
                # 初始化隐藏状态
                self._h = None
                self._c = None
                
                self._initialized = True
                logger.info("✅ Silero VAD 模型加载成功")
                
            except Exception as e:
                logger.warning(f"⚠️ Silero VAD 加载失败: {e}")
                logger.info("将回退到能量检测")
                self._initialized = True
                self._model = None
    
    def process(self, audio: bytes) -> Dict[str, Any]:
        """处理音频帧"""
        # 延迟初始化
        if not self._initialized:
            self._init_model()
        
        # 如果模型加载失败，回退到能量检测
        if self._model is None:
            return EnergyVAD(self.config).process(audio)
        
        import torch
        
        # 转换音频格式
        samples = np.frombuffer(audio, dtype=np.int16)
        
        # Silero 需要特定帧大小 (512, 1024, 1536, 2048)
        target_size = 512
        if len(samples) < target_size:
            samples = np.pad(samples, (0, target_size - len(samples)))
        elif len(samples) > target_size:
            samples = samples[:target_size]
        
        # 转换为 float32 [-1, 1]
        float_samples = samples.astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(float_samples)
        
        # 推理 - 新版 API: model(audio, sample_rate)
        with torch.no_grad():
            try:
                # 新版 Silero VAD API (v5+)
                speech_prob = self._model(audio_tensor, self.sample_rate)
                if isinstance(speech_prob, torch.Tensor):
                    prob = speech_prob.item()
                else:
                    prob = float(speech_prob)
            except Exception as e:
                # 如果失败，尝试旧版 API
                logger.debug(f"Silero API 调用失败，尝试备用方式: {e}")
                prob = 0.0
        
        self._speech_prob_history.append(prob)
        
        # 阈值判断 - 降低阈值提高灵敏度
        threshold = 0.15  # 从 0.3 再降到 0.15，提高检测率
        is_speech = prob > threshold
        
        result = {
            'is_speech': is_speech,
            'speech_prob': prob,
            'threshold': threshold,
            'speech_start': False,
            'speech_end': False,
        }
        
        # 状态机检测开始/结束
        if is_speech and not self.is_speaking:
            if len(self._speech_prob_history) >= 2:
                if list(self._speech_prob_history)[-2] > threshold:
                    self.is_speaking = True
                    result['speech_start'] = True
        elif not is_speech and self.is_speaking:
            if len(self._speech_prob_history) >= 2:
                if list(self._speech_prob_history)[-2] <= threshold:
                    self.is_speaking = False
                    result['speech_end'] = True
        
        return result
    
    def reset(self):
        """重置状态"""
        self.is_speaking = False
        self._speech_prob_history.clear()
        self._h = None
        self._c = None


class EnergyVAD(VADInterface):
    """
    能量检测 VAD
    
    简单但有效的方法，适合资源受限环境或作为备用
    """
    
    def __init__(self, config: "Config"):
        self.config = config
        self.sample_rate = config.sample_rate
        self.frame_duration_ms = 30
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
            
            # 更新噪声底
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
    
    支持：
    - 基于 Silero VAD 的打断检测（更准确）
    - 基于能量突变的打断检测（更快）
    """
    
    def __init__(self, config: "Config", use_silero: bool = True):
        self.config = config
        self.use_silero = use_silero
        
        # Silero VAD（可选）
        self._silero = None
        if use_silero:
            try:
                self._silero = SileroVAD(config)
            except:
                pass
        
        # 能量检测参数
        self.frame_duration_ms = 30
        self.sample_rate = config.sample_rate
        
        # 能量历史
        self.energy_history = deque(maxlen=50)
        self.delta_history = deque(maxlen=20)
        
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
        self.delta_threshold = 50
        
        # TTS 状态
        self._tts_playing = False
        self._lock = threading.Lock()
    
    def set_tts_state(self, playing: bool):
        """设置 TTS 播放状态"""
        with self._lock:
            self._tts_playing = playing
            if playing:
                self._frame_count = 0
                self._baseline_energy = 0
                self._speaking_frames = 0
                self._is_speaking = False
                self.energy_history.clear()
                self.delta_history.clear()
                
                if self._silero:
                    self._silero.reset()
    
    def _calc_energy(self, audio: bytes) -> float:
        """计算 RMS 能量"""
        samples = np.frombuffer(audio, dtype=np.int16)
        if len(samples) == 0:
            return 0.0
        return np.sqrt(np.mean(samples.astype(np.float32) ** 2))
    
    def process(self, audio: bytes) -> Dict[str, Any]:
        """处理音频帧，检测打断"""
        energy = self._calc_energy(audio)
        self._frame_count += 1
        
        with self._lock:
            self.energy_history.append(energy)
        
        delta = abs(energy - self._last_energy)
        self.delta_history.append(delta)
        self._last_energy = energy
        
        result = {
            'is_speech': False,
            'speech_start': False,
            'energy': energy,
            'baseline': self._baseline_energy,
            'delta': delta,
            'frame': self._frame_count,
            'reason': None,
        }
        
        # Silero VAD 模式
        if self._silero is not None:
            silero_result = self._silero.process(audio)
            result['speech_prob'] = silero_result.get('speech_prob', 0)
            
            if silero_result['is_speech']:
                result['is_speech'] = True
                self._speaking_frames += 1
                if self._speaking_frames >= self.confirm_frames:
                    if not self._is_speaking:
                        self._is_speaking = True
                        result['speech_start'] = True
                        result['reason'] = 'silero_detected'
                        logger.info(f"⚡ Silero 打断: prob={result['speech_prob']:.2f}")
            else:
                self._speaking_frames = max(0, self._speaking_frames - 1)
                if self._speaking_frames == 0:
                    self._is_speaking = False
            
            return result
        
        # 能量检测模式
        init_frames = 15
        if self._frame_count <= init_frames:
            result['reason'] = 'init'
            return result
        
        if len(self.energy_history) >= 10:
            sorted_energy = sorted(self.energy_history)
            self._baseline_energy = sorted_energy[len(sorted_energy) // 2]
        
        result['baseline'] = self._baseline_energy
        
        if self._baseline_energy > 0:
            energy_ratio = energy / self._baseline_energy
        else:
            energy_ratio = 1.0
        
        above_ratio = energy_ratio > self.energy_ratio_threshold
        above_absolute = energy > self._baseline_energy + self.min_energy
        sudden_increase = delta > self.delta_threshold and energy > self._baseline_energy
        
        is_potential_interrupt = (above_ratio and above_absolute) or sudden_increase
        
        result['energy_ratio'] = energy_ratio
        result['above_ratio'] = above_ratio
        result['above_absolute'] = above_absolute
        result['sudden_increase'] = sudden_increase
        
        if is_potential_interrupt:
            self._speaking_frames += 1
            result['is_speech'] = True
            
            if self._speaking_frames >= self.confirm_frames:
                if not self._is_speaking:
                    self._is_speaking = True
                    result['speech_start'] = True
                    result['reason'] = 'energy_spike'
                    logger.info(f"⚡ 能量打断: ratio={energy_ratio:.2f}")
        else:
            self._speaking_frames = max(0, self._speaking_frames - 1)
            if self._speaking_frames == 0:
                self._is_speaking = False
        
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
            
            if self._silero:
                self._silero.reset()


class VoiceActivityDetector:
    """
    统一的语音活动检测器
    
    默认使用 Silero VAD（更准确）
    自动回退到能量检测（如果 torch 不可用）
    """
    
    def __init__(self, config: "Config", vad_type: VADType = VADType.SILERO):
        self.config = config
        self.vad_type = vad_type
        
        # 主 VAD
        if vad_type == VADType.SILERO:
            try:
                self.normal_vad = SileroVAD(config)
                logger.info("✅ 使用 Silero VAD")
            except:
                logger.warning("⚠️ Silero VAD 不可用，回退到能量检测")
                self.normal_vad = EnergyVAD(config)
                self.vad_type = VADType.ENERGY
        else:
            self.normal_vad = EnergyVAD(config)
        
        # 打断检测器
        self.barge_in_detector = BargeInDetector(
            config, 
            use_silero=(vad_type == VADType.SILERO)
        )
        
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
        """处理音频帧"""
        with self._lock:
            tts_playing = self._tts_playing
        
        if tts_playing and self.config.barge_in_enabled:
            return self.barge_in_detector.process(audio)
        else:
            return self.normal_vad.process(audio)
    
    def process_for_interrupt(self, audio: bytes) -> Dict[str, Any]:
        """专门用于打断检测"""
        return self.barge_in_detector.process(audio)
    
    def reset(self):
        """重置所有状态"""
        self.normal_vad.reset()
        self.barge_in_detector.reset()
    
    @property
    def vad_name(self) -> str:
        """当前 VAD 名称"""
        return self.vad_type.value