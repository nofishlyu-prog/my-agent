# 🎙️ 全双工语音对话智能体

基于阿里云百练千问模型链的实时语音对话系统，支持真正的全双工对话。

## 🖥️ 跨平台支持

支持 **Windows** 和 **macOS**

## 🚀 安装

### macOS

```bash
# 1. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt
```

### Windows

```powershell
# 1. 创建虚拟环境
python -m venv .venv
.\.venv\Scripts\activate

# 2. 安装 PyAudio (Windows 需要预编译包)
# 方法 A: 使用 pipwin
pip install pipwin
pipwin install pyaudio

# 方法 B: 手动下载 whl 文件
# 从 https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio 下载对应版本
# pip install PyAudio‑0.2.14‑cp3xx‑win_amd64.whl

# 3. 安装其他依赖
pip install dashscope openai numpy torch torchaudio aiohttp
```

### 常见问题

**Q: Windows 上 PyAudio 安装失败？**
```
# 使用 conda
conda install pyaudio

# 或使用预编译包
pip install pipwin
pipwin install pyaudio
```

**Q: torch 安装太大？**
```bash
# 只安装 CPU 版本
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Q: macOS 上 PyAudio 安装失败？**
```bash
brew install portaudio
pip install pyaudio
```

## 📦 依赖

| 包 | 用途 | 必需 |
|---|---|---|
| dashscope | 阿里云 ASR/TTS | ✅ |
| openai | LLM 调用 | ✅ |
| pyaudio | 音频输入输出 | ✅ |
| numpy | 音频处理 | ✅ |
| torch | Silero VAD | 推荐 |
| torchaudio | Silero VAD | 推荐 |

## ⚙️ 配置

编辑 `config.json`：

```json
{
  "api_key": "your-api-key",
  "vad_type": "silero",
  "barge_in_enabled": true,
  ...
}
```

## 🏃 运行

```bash
# 激活虚拟环境
source .venv/bin/activate  # macOS
.\.venv\Scripts\activate   # Windows

# 运行
python main.py

# 或指定参数
python main.py --devices     # 列出音频设备
python main.py --info        # 显示系统信息
python main.py --interactive # 交互式选择设备
python main.py --debug       # 调试模式
```

## 🎤 选择音频设备

运行时选择 `3. 语音对话 (选择设备)` 可以交互式选择输入/输出设备。

或在命令行查看设备：
```bash
python main.py --devices
```

## 📖 详细文档

查看 [README.md](README.md) 获取更多信息。