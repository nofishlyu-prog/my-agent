"""
语音活动检测 (VAD) 模块

支持全双工模式下的打断检测：
- 正常模式：检测用户语音开始/结束
- 打断模式：在 TTS 播放时检测用户语音（处理声学回声）
"""

import struct
import threading
import time
import logging
from collections import deque
from typing import Dict, Any, Optional, List
import math

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    """语音活动检测器"""

    def __init__(self, config: "Config"):
        self.config = config

        # 正常 VAD 状态
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

        # ====== 打断检测专用状态 ======
        # 基线估计
        self._interrupt_baseline = 0.0
        self._interrupt_baseline_frames = 0
        self._baseline_window = deque(maxlen=30)  # 30 帧滑动窗口估计基线

        # 能量变化检测
        self._interrupt_energy_history = deque(maxlen=50)
        self._interrupt_speaking = False
        self._interrupt_frames = 0
        self._frame_count = 0

        # 动态阈值参数
        self._dynamic_threshold = 100.0
        self._last_threshold_update = 0

        # 突发检测状态
        self._burst_detected = False
        self._burst_energy_peak = 0.0
        self._consecutive_high_frames = 0
        self._consecutive_low_frames = 0

        # 用于调试的统计
        self._debug_last_log = 0

    def _calc_energy(self, audio: bytes) -> float:
        """计算音频能量（RMS）"""
        samples = struct.unpack(f'<{len(audio)//2}h', audio)
        if not samples:
            return 0.0
        return (sum(s * s for s in samples) / len(samples)) ** 0.5

    def _calc_zcr(self, audio: bytes) -> float:
        """计算零交叉率 (ZCR) - 用于区分人声和白噪声"""
        samples = struct.unpack(f'<{len(audio)//2}h', audio)
        if len(samples) < 2:
            return 0.0
        crossings = sum(1 for i in range(1, len(samples)) if samples[i] * samples[i-1] < 0)
        return crossings / len(samples)

    def _calc_spectral_centroid_indicator(self, audio: bytes) -> float:
        """
        简化的频谱质心指示器
        人声通常有更高的频谱质心，而扬声器播放的声音往往集中在低频
        这里用一个简化的计算方式
        """
        samples = struct.unpack(f'<{len(audio)//2}h', audio)
        if len(samples) < 4:
            return 0.0

        # 简单的高频能量估计
        # 通过相邻样本差分来估计高频成分
        diff_energy = sum((samples[i] - samples[i-1])**2 for i in range(1, len(samples)))
        total_energy = sum(s*s for s in samples)

        if total_energy == 0:
            return 0.0

        # 高频比例
        return math.sqrt(diff_energy / total_energy) * 100

    def set_tts_playing(self, playing: bool):
        """设置 TTS 播放状态"""
        with self._lock:
            if not playing and self.tts_playing:
                # TTS 停止时，短暂抑制检测
                self.suppress_count = 5

            self.tts_playing = playing

            # TTS 状态改变时重置打断检测状态
            if playing:
                self._reset_interrupt_state()

    def _reset_interrupt_state(self):
        """重置打断检测状态"""
        self._interrupt_baseline = 0.0
        self._interrupt_baseline_frames = 0
        self._baseline_window.clear()
        self._interrupt_energy_history.clear()
        self._interrupt_speaking = False
        self._interrupt_frames = 0
        self._frame_count = 0
        self._burst_detected = False
        self._burst_energy_peak = 0.0
        self._consecutive_high_frames = 0
        self._consecutive_low_frames = 0
        self._dynamic_threshold = 100.0

    def should_ignore(self) -> bool:
        """判断是否应该忽略当前音频（正常 VAD 使用）"""
        with self._lock:
            if self.tts_playing:
                return True
            if self.suppress_count > 0:
                self.suppress_count -= 1
                return True
            return False

    def process_for_interrupt(self, audio: bytes) -> Dict[str, Any]:
        """
        专门用于打断检测 - 改进版

        策略：
        1. 动态基线跟踪：持续估计当前背景噪声（包括 TTS 回声）
        2. 相对能量变化：检测能量相对于基线的突变
        3. 多特征融合：结合能量、ZCR、频谱特征
        4. 持续确认：需要连续多帧确认才触发打断
        """
        energy = self._calc_energy(audio)
        zcr = self._calc_zcr(audio)
        spectral = self._calc_spectral_centroid_indicator(audio)
        self._frame_count += 1

        # 记录能量历史
        self._interrupt_energy_history.append(energy)
        self._baseline_window.append(energy)

        result = {
            'is_speech': False,
            'speech_start': False,
            'energy': energy,
            'zcr': zcr,
            'spectral': spectral,
            'frame': self._frame_count,
            'phase': 'init',
            'baseline': 0.0,
            'threshold': 0.0,
            'dynamic_threshold': 0.0,
            'energy_ratio': 1.0,
            'interrupt_frames': 0,
        }

        # ====== 阶段 1：初始化基线 ======
        init_frames = getattr(self.config, 'barge_in_baseline_frames', 20)
        if self._frame_count <= init_frames:
            result['phase'] = 'init'
            return result

        # ====== 阶段 2：动态基线估计 ======
        # 使用滑动窗口的较低百分位数作为基线估计
        # 这比平均值更能抵抗突发噪声
        sorted_energy = sorted(self._baseline_window)
        baseline_percentile = 25  # 取 25% 分位数
        baseline_idx = int(len(sorted_energy) * baseline_percentile / 100)
        self._interrupt_baseline = sorted_energy[baseline_idx]

        result['baseline'] = self._interrupt_baseline

        # ====== 阶段 3：动态阈值计算 ======
        # 策略：基线 + min(基线比例, 绝对增量)
        # 在 TTS 播放时，基线通常很高（500-2000），用户说话会带来额外能量
        # 我们需要检测能量的"相对增加"，而不是绝对阈值

        min_increment = getattr(self.config, 'barge_in_min_increment', 80)
        ratio_increment = self._interrupt_baseline * 0.3  # 基线的 30%

        # 动态阈值 = 基线 + 增量
        self._dynamic_threshold = self._interrupt_baseline + max(min_increment, ratio_increment)

        result['dynamic_threshold'] = self._dynamic_threshold
        result['threshold'] = self._dynamic_threshold

        # ====== 阶段 4：能量比率检测 ======
        # 计算当前能量相对于基线的比率
        if self._interrupt_baseline > 0:
            energy_ratio = energy / self._interrupt_baseline
        else:
            energy_ratio = 1.0

        result['energy_ratio'] = energy_ratio

        # ====== 阶段 5：突发检测 + 持续确认 ======
        # 突发检测：能量比率超过阈值
        ratio_threshold = getattr(self.config, 'barge_in_ratio_threshold', 1.3)

        # 检测能量是否显著高于动态阈值
        above_threshold = energy > self._dynamic_threshold
        above_ratio = energy_ratio > ratio_threshold

        # 综合判断
        is_potential_speech = above_threshold and above_ratio

        # 持续确认机制
        confirm_frames = getattr(self.config, 'barge_in_confirm_frames', 3)

        if is_potential_speech:
            self._consecutive_high_frames += 1
            self._consecutive_low_frames = 0

            # 跟踪能量峰值
            if energy > self._burst_energy_peak:
                self._burst_energy_peak = energy

            # 确认突发
            if self._consecutive_high_frames >= confirm_frames:
                result['is_speech'] = True

                # 检测是否是新突发开始
                if not self._interrupt_speaking:
                    self._interrupt_speaking = True
                    result['speech_start'] = True
                    logger.info(
                        f"⚡ 打断触发: energy={energy:.1f}, baseline={self._interrupt_baseline:.1f}, "
                        f"ratio={energy_ratio:.2f}, zcr={zcr:.3f}, spectral={spectral:.1f}"
                    )
        else:
            self._consecutive_low_frames += 1

            # 如果连续低能量帧超过阈值，重置突发状态
            if self._consecutive_low_frames >= 3:
                self._consecutive_high_frames = 0
                self._burst_energy_peak = 0.0
                self._interrupt_speaking = False

        result['phase'] = 'detect'
        result['interrupt_frames'] = self._consecutive_high_frames

        # 定期输出调试日志（每 30 帧，约 300ms）
        current_time = time.time()
        if current_time - self._debug_last_log > 0.3 or result['speech_start']:
            self._debug_last_log = current_time
            logger.debug(
                f"[VAD] frame={self._frame_count}, energy={energy:.1f}, "
                f"baseline={self._interrupt_baseline:.1f}, ratio={energy_ratio:.2f}, "
                f"threshold={self._dynamic_threshold:.1f}, is_speech={result['is_speech']}, "
                f"high_frames={self._consecutive_high_frames}"
            )

        return result

    def process(self, audio: bytes) -> Dict[str, Any]:
        """处理音频帧，检测语音活动（正常模式）"""
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
        """重置检测器状态"""
        self.is_speaking = False
        self.silence_start = None
        self.speech_frames = 0
        self.silence_count = 0
        with self._lock:
            self.tts_playing = False
            self.suppress_count = 0
        self._reset_interrupt_state()