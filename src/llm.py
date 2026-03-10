"""
大语言模型 (LLM) 模块
"""

import threading
import logging
from typing import List, Dict

try:
    from openai import OpenAI
except ImportError as e:
    raise ImportError("请安装 openai: pip install openai") from e

logger = logging.getLogger(__name__)


class LanguageModel:
    """大语言模型接口"""

    def __init__(self, config: "Config"):
        self.config = config
        self.conversation_history: List[Dict[str, str]] = []
        self._lock = threading.Lock()

        self.client = OpenAI(
            api_key=config.api_key,
            base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
        )

    def chat(self, user_input: str) -> str:
        """发送对话请求（同步版本）"""
        import asyncio

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

    async def chat_async(self, user_input: str) -> str:
        """发送对话请求（异步版本）"""
        return self.chat(user_input)

    def clear_history(self):
        """清空对话历史"""
        with self._lock:
            self.conversation_history.clear()