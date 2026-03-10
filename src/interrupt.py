"""
语义打断检测模块
"""

import re
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class SemanticInterruptDetector:
    """语义打断检测器"""

    def __init__(self, config: "Config"):
        self.keywords: List[str] = config.interrupt_keywords
        self.interrupt_pattern = re.compile('|'.join(re.escape(k) for k in self.keywords))

    def check(self, text: str) -> Dict[str, Any]:
        """检查文本是否包含打断关键词"""
        if not text or not text.strip():
            return {'interrupt': False, 'keyword': None}
        match = self.interrupt_pattern.search(text)
        if match:
            return {'interrupt': True, 'keyword': match.group()}
        return {'interrupt': False, 'keyword': None}

    def add_keyword(self, keyword: str):
        """添加打断关键词"""
        if keyword and keyword not in self.keywords:
            self.keywords.append(keyword)
            self.interrupt_pattern = re.compile('|'.join(re.escape(k) for k in self.keywords))

    def remove_keyword(self, keyword: str):
        """移除打断关键词"""
        if keyword in self.keywords:
            self.keywords.remove(keyword)
            self.interrupt_pattern = re.compile('|'.join(re.escape(k) for k in self.keywords))