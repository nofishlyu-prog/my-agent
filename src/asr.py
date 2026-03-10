"""
语音识别 (ASR) 模块
"""

import queue
import threading
import time
import logging
from typing import Optional

try:
    import dashscope
    from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult
except ImportError as e:
    raise ImportError("请安装 dashscope: pip install dashscope") from e

logger = logging.getLogger(__name__)


class SpeechRecognizer:
    """语音识别器"""

    def __init__(self, config: "Config"):
        self.config = config
        dashscope.api_key = config.api_key

        self.result_queue = queue.Queue()
        self._current_text = ""
        self.recognizer = None
        self.is_connected = False
        self._lock = threading.Lock()
        self._callback_ref = None
        self._partial_text = ""
        self._audio_buffer = bytearray()

    def _create_callback(self):
        """创建 ASR 回调"""
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
        """启动 ASR"""
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
        """发送音频数据"""
        if self.is_connected and self.recognizer:
            self.recognizer.send_audio_frame(audio)

    def stop(self):
        """停止 ASR"""
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
        """获取识别结果"""
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
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except:
                break