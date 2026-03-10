#!/usr/bin/env python3
"""
全双工语音对话智能体 - 主程序入口
"""

import asyncio
import logging
import sys
import argparse
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src import (
    Config,
    FullDuplexAgent,
    LanguageModel,
    TextToSpeech,
    SemanticInterruptDetector
)

# 配置日志
def setup_logging(debug: bool = False):
    """设置日志"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    # 降低第三方库日志级别
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('dashscope').setLevel(logging.WARNING)


class TestMode:
    """测试模式"""

    def __init__(self, config: Config):
        self.config = config
        self.llm = LanguageModel(config)
        self.tts = TextToSpeech(config)
        self.interrupt_detector = SemanticInterruptDetector(config)

    def _play_audio(self, text: str):
        """播放音频"""
        import pyaudio

        audio = self.tts.synthesize(text)
        if not audio:
            return
        try:
            p = pyaudio.PyAudio()
            stream = p.open(
                rate=self.config.tts_sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                output=True
            )
            stream.write(audio[44:])  # 跳过 WAV 头
            stream.stop_stream()
            stream.close()
            p.terminate()
        except Exception as e:
            logging.error(f"播放错误：{e}")

    async def run(self):
        """运行测试模式"""
        print("=" * 60)
        print("🧪 测试模式")
        print("=" * 60)

        # 语义打断测试
        print("\n📊 语义打断测试：")
        test_cases = [
            ("等等，我想问一下", True),
            ("嗯，那个...", False),
            ("停下，不对", True),
            ("今天天气不错", False),
            ("闭嘴", True),
            ("重新说一遍", True),
        ]

        passed = 0
        for text, expected in test_cases:
            result = self.interrupt_detector.check(text)
            is_correct = result['interrupt'] == expected
            if is_correct:
                passed += 1
            status = "✅" if is_correct else "❌"
            action = "⚡ 打断" if result['interrupt'] else "✓ 不打断"
            print(f"  {status} {action}: \"{text}\"")

        print(f"\n测试通过：{passed}/{len(test_cases)}")

        self._play_audio("你好，我是你的语音助手小智")
        print("\n💬 文字对话模式，输入 quit 退出\n")

        while True:
            try:
                text = input("你：").strip()
                if text.lower() == 'quit':
                    break
                if not text:
                    continue

                result = self.interrupt_detector.check(text)
                if result['interrupt']:
                    print(f"⚡ 检测到打断关键词：{result['keyword']}")
                    self._play_audio("好的，请问有什么事？")
                    continue

                print("助手：", end='', flush=True)
                response = self.llm.chat(text)
                print(response)
                self._play_audio(response)
            except KeyboardInterrupt:
                break
        print("\n👋 再见！")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='全双工语音对话智能体')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--test', action='store_true', help='运行测试模式')
    args = parser.parse_args()
    
    setup_logging(args.debug)
    
    config = Config.from_json("config.json")

    if args.test:
        await TestMode(config).run()
    else:
        print("\n请选择模式：")
        print("1. 测试模式 (文字对话)")
        print("2. 语音对话")
        print()
        
        try:
            choice = input("选择 [1/2]: ").strip()
        except EOFError:
            choice = '2'
        
        if choice == '1':
            await TestMode(config).run()
        else:
            await FullDuplexAgent(config).run()


if __name__ == '__main__':
    asyncio.run(main())