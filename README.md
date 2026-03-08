# 🎙️ 全双工语音对话智能体 (Qwen3-Omni)

基于阿里云百练千问模型链的实时语音对话系统，支持 **Qwen3-Omni** 模型，实现真正的全双工对话。

## 🏗️ 系统架构

```
┌────────────────────────────────────────────────────────────┐
│              全双工语音对话智能体                              │
│                                                            │
│  ┌──────┐   ┌─────┐   ┌─────┐   ┌──────────┐   ┌─────┐   │
│  │麦克风 │──→│ VAD │──→│ ASR │──→│ LLM      │──→│ TTS │──→│
│  └──────┘   └─────┘   └─────┘   │ Qwen3    │   └─────┘   │
│       ↑         │                           ↑            │
│       └─────────┴────── Barge-in ───────────┘            │
│                                                            │
│  模型链：FSMN-VAD → Paraformer → Qwen3-Omni → CosyVoice   │
└────────────────────────────────────────────────────────────┘
```

## 📦 模型链

| 模块 | 模型 | 说明 |
|------|------|------|
| **VAD** | 能量检测 + ZCR | 语音活动检测，动态阈值 |
| **ASR** | paraformer-realtime-v2 | 阿里云实时语音识别 |
| **LLM** | qwen3-omni-flash-2025-12-01 | Qwen3-Omni-Flash (OpenAI 兼容接口) |
| **TTS** | cosyvoice-v1 | 阿里云语音合成 |

## 🚀 快速开始

### 1. 安装依赖

```bash
cd ~/Desktop/voice-agent
pip install -r requirements.txt
```

**依赖说明**：
- `dashscope` - 阿里云 ASR/TTS
- `openai` - Qwen3-Omni LLM 调用
- `pyaudio` - 音频输入输出

### 2. 配置 API Key

编辑 `config.json`，修改为你的 API Key：

```json
{
  "api_key": "sk-your-api-key"
}
```

### 3. 运行

```bash
python3 voice_agent.py
```

### 4. 选择模式

- **模式 1**: 测试模式（文字对话 + 语音播放）
- **模式 2**: 语音对话（全双工实时对话）

## ✨ 核心功能

### 全双工对话
- **边听边说**：支持 Barge-in（抢话）功能
- **随时打断**：播放中可随时打断
- **低延迟**：首 token 响应约 1-2 秒

### 多轮对话
- 保持对话历史（默认 10 轮）
- 上下文理解
- 流式响应

### 语义打断
支持打断关键词：
```
等等、等一下、停下、停一下、停
不要说了、别说、别说了
不对、错了、不是这样、不是
取消、停止、闭嘴、打断、慢点、重新说
喂、你好、在吗、请问
```

## ⚙️ 配置

编辑 `config.json`：

```json
{
  "api_key": "sk-your-api-key",
  "llm_model": "qwen3-omni-flash-2025-12-01",
  "vad_threshold": 300,
  "vad_silence_ms": 500,
  "asr_model": "paraformer-realtime-v2",
  "tts_model": "cosyvoice-v1",
  "tts_voice": "longxiaochun",
  "enable_full_duplex": true,
  "barge_in_enabled": true,
  "first_token_timeout": 2.0,
  "interrupt_keywords": [
    "等等", "等一下", "停下", "停一下", "停",
    "不要说了", "别说", "别说了",
    "不对", "错了", "不是这样", "不是",
    "取消", "停止", "闭嘴", "打断", "慢点", "重新说"
  ]
}
```

### 配置项说明

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `api_key` | 阿里云 API Key | - |
| `llm_model` | LLM 模型 | `qwen-omni-turbo` |
| `vad_threshold` | VAD 能量阈值 | `300` |
| `vad_silence_ms` | 沉默检测时间 (ms) | `500` |
| `asr_model` | ASR 模型 | `paraformer-realtime-v2` |
| `tts_voice` | TTS 音色 | `longxiaochun` |
| `enable_full_duplex` | 全双工模式 | `true` |
| `barge_in_enabled` | 抢话功能 | `true` |
| `first_token_timeout` | 首 token 超时 (s) | `2.0` |

## 🎯 使用技巧

### 说话技巧
- **清晰**：普通话，语速适中
- **停顿**：说完停顿 0.5 秒等待识别
- **简短**：问题简短，回复更快

### 打断技巧
播放时说：
- "等等" → 立即停止，开始新对话
- "停下" → 同上
- "不对" → 打断并纠正
- 其他内容 → 不打断，继续播放

### 调优建议

| 问题 | 解决方案 |
|------|---------|
| 识别不灵敏 | 降低 `vad_threshold` 到 250 |
| 噪声误触发 | 提高到 400-500 |
| 打断不工作 | 检查 `interrupt_keywords` |
| 回复太慢 | 使用 `qwen-omni-turbo` 模型 |
| TTS 太慢 | 缩短 `first_token_timeout` |

## 📁 项目结构

```
voice-agent/
├── voice_agent.py    # 主程序
├── config.json       # 配置文件
├── requirements.txt  # 依赖
├── README.md         # 本文档
└── __init__.py       # 包初始化
```

## 🔧 常见问题

**Q: 无法识别语音？**
A: 检查 API Key、网络连接、麦克风权限

**Q: 打断不生效？**
A: 确保说出完整的打断关键词

**Q: 回复太慢？**
A: 使用 qwen-omni-turbo 模型，检查网络延迟

**Q: 如何切换模型？**
A: 修改 `config.json` 中的 `llm_model` 字段

## 📊 性能指标

| 指标 | 目标值 |
|------|--------|
| 首 token 延迟 | < 2s |
| 打断响应时间 | < 0.5s |
| 语音识别准确率 | > 95% |

## 📖 参考

- [阿里云百练文档](https://help.aliyun.com/zh/model-studio/)
- [Qwen3-Omni 模型](https://github.com/QwenLM/Qwen)
- [Paraformer 模型](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch)
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
