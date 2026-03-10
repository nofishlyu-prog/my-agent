"""
回声消除模块 (AEC - Acoustic Echo Cancellation)

实现基于参考信号的自适应滤波器：
1. 记录 TTS 输出的参考信号
2. 使用 NLMS 自适应滤波器估计回声路径
3. 从麦克风输入中减去估计的回声

原理：
- 回声 = H * x(n)，其中 H 是房间脉冲响应，x(n) 是 TTS 输出
- NLMS 滤波器学习 H 的估计
- 输出 = d(n) - H_est * x(n)，其中 d(n) 是麦克风输入
"""

import numpy as np
from collections import deque
import logging

logger = logging.getLogger(__name__)


class NLMSFilter:
    """
    归一化最小均方 (NLMS) 自适应滤波器
    
    用于估计和消除回声
    """
    
    def __init__(self, filter_length: int = 1024, step_size: float = 0.5, regularization: float = 1e-6):
        """
        初始化 NLMS 滤波器
        
        Args:
            filter_length: 滤波器长度（采样点数），越长效果越好但延迟越大
            step_size: 步长参数，控制收敛速度和稳态误差的权衡
            regularization: 正则化参数，防止除零
        """
        self.filter_length = filter_length
        self.step_size = step_size
        self.regularization = regularization
        
        # 滤波器系数
        self.weights = np.zeros(filter_length, dtype=np.float32)
        
        # 参考信号缓冲
        self.reference_buffer = deque(maxlen=filter_length)
        
        # 初始化缓冲区
        for _ in range(filter_length):
            self.reference_buffer.append(0.0)
    
    def process(self, microphone_sample: float, reference_sample: float) -> float:
        """
        处理一个采样点
        
        Args:
            microphone_sample: 麦克风输入采样点
            reference_sample: 参考信号（TTS 输出）采样点
            
        Returns:
            消除回声后的信号
        """
        # 更新参考信号缓冲
        self.reference_buffer.append(reference_sample)
        
        # 获取参考信号向量
        x = np.array(self.reference_buffer, dtype=np.float32)
        
        # 计算估计的回声
        echo_estimate = np.dot(self.weights, x)
        
        # 计算误差（消除回声后的信号）
        error = microphone_sample - echo_estimate
        
        # 计算参考信号的功率
        x_power = np.dot(x, x) + self.regularization
        
        # NLMS 权重更新
        self.weights += self.step_size * error * x / x_power
        
        return error
    
    def process_frame(self, mic_frame: np.ndarray, ref_frame: np.ndarray) -> np.ndarray:
        """
        处理一帧音频
        
        Args:
            mic_frame: 麦克风输入帧 (float32)
            ref_frame: 参考信号帧 (float32)
            
        Returns:
            消除回声后的帧
        """
        output = np.zeros(len(mic_frame), dtype=np.float32)
        
        for i in range(len(mic_frame)):
            output[i] = self.process(mic_frame[i], ref_frame[i])
        
        return output
    
    def reset(self):
        """重置滤波器"""
        self.weights = np.zeros(self.filter_length, dtype=np.float32)
        self.reference_buffer.clear()
        for _ in range(self.filter_length):
            self.reference_buffer.append(0.0)


class AECProcessor:
    """
    回声消除处理器
    
    管理参考信号缓冲和 NLMS 滤波器
    """
    
    def __init__(self, sample_rate: int = 16000, filter_length_ms: int = 64):
        """
        初始化 AEC 处理器
        
        Args:
            sample_rate: 采样率
            filter_length_ms: 滤波器长度（毫秒）
        """
        self.sample_rate = sample_rate
        self.filter_length = int(sample_rate * filter_length_ms / 1000)
        
        # NLMS 滤波器
        self.nlms = NLMSFilter(
            filter_length=self.filter_length,
            step_size=0.3,
            regularization=1e-6
        )
        
        # TTS 参考信号缓冲
        # 需要足够大以处理系统延迟
        self.tts_buffer = deque(maxlen=self.filter_length * 4)
        
        # 延迟估计（系统延迟，需要校准）
        self.system_delay_samples = int(sample_rate * 0.05)  # 默认 50ms
        
        # 状态
        self._tts_playing = False
        self._tts_reference = bytearray()
        
        logger.info(f"AEC 初始化: filter_length={self.filter_length}, delay={self.system_delay_samples}")
    
    def start_tts(self):
        """TTS 播放开始"""
        self._tts_playing = True
        self._tts_reference = bytearray()
    
    def stop_tts(self):
        """TTS 播放结束"""
        self._tts_playing = False
    
    def add_tts_reference(self, audio: bytes):
        """
        添加 TTS 输出作为参考信号
        
        Args:
            audio: TTS 输出的音频 (PCM int16)
        """
        # 转换为 float32
        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        
        # 添加到缓冲区
        for s in samples:
            self.tts_buffer.append(s)
    
    def process_mic_input(self, audio: bytes) -> bytes:
        """
        处理麦克风输入，消除回声
        
        Args:
            audio: 麦克风输入 (PCM int16)
            
        Returns:
            消除回声后的音频 (PCM int16)
        """
        if not self._tts_playing or len(self.tts_buffer) < self.filter_length:
            # TTS 未播放或参考信号不足，直接返回原信号
            return audio
        
        # 转换为 float32
        mic_samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        
        # 获取对齐的参考信号
        output_samples = np.zeros(len(mic_samples), dtype=np.float32)
        
        for i in range(len(mic_samples)):
            # 从缓冲区获取参考信号（考虑延迟）
            if len(self.tts_buffer) > self.system_delay_samples + i:
                ref_sample = self.tts_buffer[-(self.system_delay_samples + len(mic_samples) - i)]
            else:
                ref_sample = 0.0
            
            # NLMS 处理
            output_samples[i] = self.nlms.process(mic_samples[i], ref_sample)
        
        # 转换回 int16
        # 应用软限制防止削波
        output_samples = np.clip(output_samples, -0.99, 0.99)
        output_int16 = (output_samples * 32767).astype(np.int16)
        
        return output_int16.tobytes()
    
    def reset(self):
        """重置处理器"""
        self.nlms.reset()
        self.tts_buffer.clear()
        self._tts_playing = False
        self._tts_reference = bytearray()


class SimpleAEC:
    """
    简化的回声消除器
    
    使用频谱减法，计算量更小但效果也不错
    """
    
    def __init__(self, sample_rate: int = 16000, echo_suppression: float = 0.7):
        """
        初始化
        
        Args:
            sample_rate: 采样率
            echo_suppression: 回声抑制系数 (0-1)
        """
        self.sample_rate = sample_rate
        self.echo_suppression = echo_suppression
        
        # 能量跟踪
        self.tts_energy = 0.0
        self.tts_energy_alpha = 0.1  # 平滑系数
        
        # 频谱缓冲
        self.fft_size = 512
        
        logger.info(f"SimpleAEC 初始化: suppression={echo_suppression}")
    
    def update_tts_energy(self, audio: bytes):
        """更新 TTS 能量估计"""
        samples = np.frombuffer(audio, dtype=np.int16)
        energy = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
        
        # 平滑更新
        self.tts_energy = self.tts_energy_alpha * energy + (1 - self.tts_energy_alpha) * self.tts_energy
    
    def suppress_echo(self, mic_audio: bytes, tts_audio: bytes = None) -> bytes:
        """
        抑制回声
        
        Args:
            mic_audio: 麦克风输入
            tts_audio: TTS 输出（可选，用于更新能量）
            
        Returns:
            抑制回声后的音频
        """
        if tts_audio:
            self.update_tts_energy(tts_audio)
        
        if self.tts_energy < 10:
            # TTS 能量很低，不处理
            return mic_audio
        
        # 转换
        mic_samples = np.frombuffer(mic_audio, dtype=np.int16).astype(np.float32)
        
        # 计算抑制因子
        # 如果麦克风能量接近 TTS 能量，说明主要是回声
        mic_energy = np.sqrt(np.mean(mic_samples ** 2))
        
        if mic_energy > 0:
            # 估计回声比例
            echo_ratio = min(1.0, self.tts_energy / mic_energy)
            
            # 抑制回声
            suppression = 1.0 - echo_ratio * self.echo_suppression
            suppression = max(0.1, suppression)  # 至少保留 10%
            
            mic_samples = mic_samples * suppression
        
        # 转换回 int16
        mic_samples = np.clip(mic_samples, -32767, 32767)
        return mic_samples.astype(np.int16).tobytes()
    
    def reset(self):
        """重置"""
        self.tts_energy = 0.0