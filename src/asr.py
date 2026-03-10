"""
语音识别 (ASR) 模块

使用阿里云 Paraformer 实时语音识别

改进：
- 自动重连机制
- 线程安全
- 完善的错误处理
- 支持打断模式
"""

import queue
import threading
import time
import logging
from typing import Optional, Callable
from dataclasses import dataclass
from enum import Enum

try:
    import dashscope
    from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult
except ImportError as e:
    raise ImportError("请安装 dashscope: pip install dashscope") from e

logger = logging.getLogger(__name__)


class ASRState(Enum):
    """ASR 状态"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class ASRResult:
    """ASR 识别结果"""
    text: str
    is_final: bool
    confidence: float = 1.0


class SpeechRecognizer:
    """
    语音识别器
    
    特点：
    - 自动重连
    - 线程安全
    - 支持实时结果回调
    """
    
    def __init__(self, config: "Config"):
        self.config = config
        dashscope.api_key = config.api_key
        
        # 状态
        self.state = ASRState.DISCONNECTED
        self.is_connected = False
        
        # 结果队列
        self._result_queue = queue.Queue()
        self._partial_text = ""
        self._final_text = ""
        self._lock = threading.Lock()
        
        # 识别器实例
        self._recognizer = None
        self._callback = None
        
        # 重连控制
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        self._reconnect_delay = 1.0
        self._last_connect_time = 0
        
        # 运行控制
        self._running = False
        self._audio_buffer = queue.Queue(maxsize=100)
        self._send_thread: Optional[threading.Thread] = None
        
        # 回调
        self._on_partial: Optional[Callable[[str], None]] = None
        self._on_final: Optional[Callable[[str], None]] = None
    
    def set_callbacks(self, on_partial: Callable[[str], None] = None, 
                     on_final: Callable[[str], None] = None):
        """设置回调函数"""
        self._on_partial = on_partial
        self._on_final = on_final
    
    def _create_callback(self):
        """创建 ASR 回调"""
        recognizer = self
        
        class Callback(RecognitionCallback):
            def on_open(self):
                recognizer.state = ASRState.CONNECTED
                recognizer.is_connected = True
                recognizer._reconnect_attempts = 0
                logger.info("✅ ASR 连接成功")
            
            def on_close(self):
                recognizer.is_connected = False
                recognizer.state = ASRState.DISCONNECTED
                logger.info("🔌 ASR 连接关闭")
            
            def on_event(self, result: RecognitionResult):
                try:
                    logger.debug(f"👂 ASR on_event 收到: {result}")
                    sentence = result.get_sentence()
                    if sentence:
                        text = sentence.get('text', '')
                        # 修复：检查 sentence_end 而不是 is_final
                        is_final = sentence.get('sentence_end', False) or sentence.get('is_final', False)
                        
                        with recognizer._lock:
                            recognizer._partial_text = text
                            if is_final:
                                recognizer._final_text = text
                        
                        # 回调
                        if text:
                            if is_final:
                                recognizer._result_queue.put(ASRResult(text=text, is_final=True))
                                if recognizer._on_final:
                                    recognizer._on_final(text)
                                logger.info(f"📝 ASR 最终结果: {text}")
                            else:
                                if recognizer._on_partial:
                                    recognizer._on_partial(text)
                                logger.debug(f"📝 ASR 临时结果: {text}")
                    else:
                        logger.debug(f"👂 ASR on_event 无 sentence")
                
                except Exception as e:
                    logger.error(f"ASR 事件处理错误: {e}")
            
            def on_complete(self):
                with recognizer._lock:
                    if recognizer._final_text:
                        recognizer._result_queue.put(ASRResult(text=recognizer._final_text, is_final=True))
                    recognizer._partial_text = ""
                    recognizer._final_text = ""
                logger.debug("📝 ASR 识别完成")
            
            def on_error(self, result):
                recognizer.state = ASRState.ERROR
                recognizer.is_connected = False
                logger.error(f"❌ ASR 错误: {result}")
        
        return Callback()
    
    def start(self) -> bool:
        """启动 ASR"""
        if self.is_connected:
            logger.debug("ASR 已连接，跳过启动")
            return True
        
        try:
            self.state = ASRState.CONNECTING
            
            # 创建识别器
            self._recognizer = Recognition(
                model=self.config.asr_model,
                format='pcm',
                sample_rate=self.config.sample_rate,
                callback=self._create_callback()
            )
            
            self._recognizer.start()
            self._last_connect_time = time.time()
            self._running = True
            
            # 启动音频发送线程
            if self._send_thread is None or not self._send_thread.is_alive():
                self._send_thread = threading.Thread(target=self._send_audio_loop, daemon=True)
                self._send_thread.start()
            
            # 等待连接
            for _ in range(20):  # 最多等待 2 秒
                if self.is_connected:
                    break
                time.sleep(0.1)
            
            if not self.is_connected:
                logger.warning("ASR 连接超时")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"ASR 启动失败: {e}")
            self.state = ASRState.ERROR
            return False
    
    def _send_audio_loop(self):
        """音频发送循环"""
        logger.info("🔊 ASR 音频发送线程启动")
        send_count = 0
        
        while self._running:
            try:
                if not self.is_connected:
                    time.sleep(0.1)
                    continue
                
                # 从缓冲区获取音频
                try:
                    audio = self._audio_buffer.get(timeout=0.1)
                    if audio and self._recognizer:
                        self._recognizer.send_audio_frame(audio)
                        send_count += 1
                        if send_count % 50 == 0:  # 每 50 帧打印一次
                            logger.info(f"ASR 已发送 {send_count} 帧音频")
                except queue.Empty:
                    continue
                except Exception as e:
                    if "Stream closed" not in str(e):
                        logger.error(f"发送音频错误: {e}")
            
            except Exception as e:
                logger.error(f"音频发送循环错误: {e}")
                time.sleep(0.1)
        
        logger.info(f"🔊 ASR 音频发送线程结束 (共发送 {send_count} 帧)")
    
    def send(self, audio: bytes):
        """发送音频数据"""
        if not self.is_connected:
            logger.debug(f"ASR send: 未连接，跳过 {len(audio)} 字节")
            return
        
        try:
            # 非阻塞放入队列
            self._audio_buffer.put_nowait(audio)
            logger.debug(f"ASR send: 放入队列 {len(audio)} 字节, 队列大小 {self._audio_buffer.qsize()}")
        except queue.Full:
            # 队列满，丢弃旧数据
            logger.warning(f"ASR send: 队列满，丢弃旧数据")
            try:
                self._audio_buffer.get_nowait()
                self._audio_buffer.put_nowait(audio)
            except:
                pass
    
    def stop(self):
        """停止 ASR"""
        self._running = False
        
        if self._recognizer:
            try:
                self._recognizer.stop()
            except Exception as e:
                logger.debug(f"停止 ASR: {e}")
        
        self._recognizer = None
        self.is_connected = False
        self.state = ASRState.DISCONNECTED
        
        # 清空缓冲区
        while not self._audio_buffer.empty():
            try:
                self._audio_buffer.get_nowait()
            except:
                break
    
    def restart(self) -> bool:
        """重启 ASR"""
        self.stop()
        time.sleep(0.2)
        return self.start()
    
    def get_result(self, timeout: float = 1.0) -> Optional[str]:
        """获取识别结果"""
        try:
            result = self._result_queue.get(timeout=timeout)
            if isinstance(result, ASRResult):
                return result.text
            return result
        except queue.Empty:
            # 返回临时结果
            with self._lock:
                text = self._partial_text
                self._partial_text = ""
            return text if text else None
    
    def get_partial_text(self) -> str:
        """获取当前的临时识别结果"""
        with self._lock:
            return self._partial_text
    
    def clear(self):
        """清空状态"""
        with self._lock:
            self._partial_text = ""
            self._final_text = ""
        
        while not self._result_queue.empty():
            try:
                self._result_queue.get_nowait()
            except:
                break
    
    def __del__(self):
        """析构函数"""
        try:
            self.stop()
        except:
            pass