"""
状态机模块
"""

from enum import Enum


class AgentState(Enum):
    """智能体状态"""
    IDLE = "空闲"
    LISTENING = "聆听"
    THINKING = "思考"
    SPEAKING = "说话"
    INTERRUPTED = "被打断"