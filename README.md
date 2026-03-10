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
| **VAD** | 动态能量检测 + ZCR | 语音活动检测，支持全双工打断 |
| **ASR** | paraformer-realtime-v2 | 阿里云实时语音识别 |
| **LLM** | qwen3-omni-flash-2025-12-01 | Qwen3-Omni-Flash (OpenAI 兼容接口) |
| **TTS** | cosyvoice-v1 | 阿里云语音合成 |

## 🚀 快速开始

### 1. 安装依赖

```bash
cd my-agent
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
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
python3 main.py
```

### 4. 选择模式

- **模式 1**: 测试模式（文字对话 + 语音播放）
- **模式 2**: 语音对话（全双工实时对话）

## ✨ 核心功能

### 全双工对话
- **边听边说**：TTS 播放时持续检测用户语音
- **即时打断**：检测到用户说话立即停止 TTS
- **动态基线**：自适应背景噪声，无需手动调整阈值

### 打断检测策略（v2.0 更新）

打断检测采用 **动态能量变化检测** 策略：

1. **动态基线估计**：使用滑动窗口的 25% 分位数作为基线
2. **相对能量检测**：检测能量相对于基线的突变（需要高出 30%）
3. **双阈值确认**：
   - 绝对增量阈值：min_increment（默认 80）
   - 相对比率阈值：ratio_threshold（默认 1.3）
4. **持续确认机制**：需要连续 3 帧确认才触发打断

```
能量 │         用户说话
     │        ╱╲
     │       ╱  ╲
     │──────╱────╲──── 阈值 = 基线 + 增量
     │     ╱      ╲
     │────╱────────╲── 基线（TTS 回声）
     └─────────────────→ 时间
```

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
```

## ⚙️ 配置

编辑 `config.json`：

```json
{
  "api_key": "sk-your-api-key",
  "llm_model": "qwen3-omni-flash-2025-12-01",
  "vad_threshold": 100,
  "vad_silence_ms": 500,
  "asr_model": "paraformer-realtime-v2",
  "tts_model": "cosyvoice-v1",
  "tts_voice": "longxiaochun",
  "enable_full_duplex": true,
  "barge_in_enabled": true,
  "barge_in_baseline_frames": 20,
  "barge_in_confirm_frames": 3,
  "barge_in_min_increment": 80,
  "barge_in_ratio_threshold": 1.3,
  "interrupt_keywords": [...]
}
```

### 打断检测参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `barge_in_baseline_frames` | 基线估计帧数 | 20 (~200ms) |
| `barge_in_confirm_frames` | 确认帧数 | 3 |
| `barge_in_min_increment` | 最小能量增量 | 80 |
| `barge_in_ratio_threshold` | 能量比率阈值 | 1.3 (高 30%) |

### 调优建议

| 问题 | 解决方案 |
|------|---------|
| 打断太灵敏（误触发） | 提高 `barge_in_ratio_threshold` 到 1.5 |
| 打断不灵敏（需要大声喊） | 降低 `barge_in_min_increment` 到 50 |
| 打断反应慢 | 减少 `barge_in_confirm_frames` 到 2 |
| 使用耳机时 | 可以降低所有阈值（无回声干扰）|

## 🎯 使用技巧

### 说话技巧
- **清晰**：普通话，语速适中
- **停顿**：说完停顿 0.5 秒等待识别
- **简短**：问题简短，回复更快

### 打断技巧
- **声音足够大**：说话需要比 TTS 背景噪声高 30%
- **清晰发音**：有助于能量检测
- **持续说话**：需要连续 3 帧（约 30ms）确认

### 最佳实践
- **使用耳机**：消除声学回声，打断检测更准确
- **安静环境**：减少背景噪声干扰
- **靠近麦克风**：提高信噪比

## 📁 项目结构

```
my-agent/
├── main.py            # 主程序入口
├── config.json        # 配置文件
├── requirements.txt   # 依赖
├── README.md          # 本文档
└── src/
    ├── __init__.py    # 包入口
    ├── config.py      # 配置管理
    ├── state.py       # 状态机
    ├── vad.py         # VAD 模块（支持全双工）
    ├── asr.py         # ASR 模块
    ├── llm.py         # LLM 模块
    ├── tts.py         # TTS 模块
    ├── interrupt.py   # 语义打断检测
    └── agent.py       # 全双工智能体
```

## 🔧 常见问题

**Q: 打断不生效？**
A: 
1. 确保说话声音足够大（超过背景噪声 30%）
2. 检查 `barge_in_enabled` 为 `true`
3. 尝试使用耳机减少回声
4. 降低 `barge_in_ratio_threshold` 到 1.2

**Q: 打断太灵敏，误触发？**
A: 
1. 提高 `barge_in_ratio_threshold` 到 1.5
2. 增加 `barge_in_confirm_frames` 到 4
3. 提高 `barge_in_min_increment` 到 100

**Q: 使用耳机时打断更好用？**
A: 是的，耳机消除了扬声器→麦克风的回声，使打断检测更准确

**Q: 无法识别语音？**
A: 检查 API Key、网络连接、麦克风权限

## 📊 性能指标

| 指标 | 目标值 |
|------|--------|
| 首 token 延迟 | < 2s |
| 打断响应时间 | < 100ms |
| 打断检测准确率 | > 90%（使用耳机时）|

## 📖 参考

- [阿里云百练文档](https://help.aliyun.com/zh/model-studio/)
- [Qwen3-Omni 模型](https://github.com/QwenLM/Qwen)
- [Paraformer 模型](https://www.modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch)

## 📝 更新日志

### v2.0 (2026-03-10)
- 重写打断检测算法，采用动态能量变化检测
- 添加多特征融合（能量 + ZCR + 频谱特征）
- 添加可配置的打断检测参数
- 改进线程同步和 ASR 管理
- 修复 TTS 播放时无法打断的问题

### v1.0 (初始版本)
- 基础全双工语音对话
- 基于关键词的语义打断

### v1.0 (2026-03-09)
- 初始版本
- 基本的全双工对话功能
- 模块化架构