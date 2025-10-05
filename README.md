
# FreeSWITCH ESL Python 示例

这是一个基于 Python 的 FreeSWITCH ESL (Event Socket Library) 示例项目，展示了如何使用 Python 与 FreeSWITCH 进行通信，实现呼叫处理和事件监听功能。

## 🚀 功能特性

- **Inbound 应用**: 客户端模式连接 FreeSWITCH，监听呼叫事件
- **Outbound 应用**: 服务器模式处理 FreeSWITCH 发起的呼叫
- **智能呼叫处理**: 支持黑名单过滤、音频播放、呼叫桥接
- **多层音频备用**: 主音频文件 → 备用音频 → 音调生成
- **详细日志记录**: 完整的调试和监控信息
- **模块化设计**: 清晰的代码结构，易于维护和扩展

## 📋 先决条件

- Python 3.7+
- FreeSWITCH 服务器 (已安装并运行)
- 网络连接 (FreeSWITCH 和 Python 应用之间)

## 🛠️ 安装

1. **克隆仓库**
   ```bash
   git clone git@github.com:dongjb741280/freeswitch-esl-python-sample.git
   cd freeswitch-esl-python-sample
   ```

2. **创建虚拟环境** (推荐)
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 或
   venv\Scripts\activate     # Windows
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

## 📖 使用方法

### Inbound 应用 (事件监听)

Inbound 应用以客户端方式连接到 FreeSWITCH，监听呼叫事件并显示详细信息。

**启动命令:**
```bash
python inbound_app.py
```

**功能说明:**
- 连接到 FreeSWITCH 事件系统
- 监听所有通道事件 (CHANNEL_CREATE, CHANNEL_BRIDGE, CHANNEL_ANSWER, CHANNEL_HANGUP)
- 显示呼叫的详细信息，包括主叫号码、被叫号码、挂断原因等

### AI 呼叫中心 (FunASR 实时识别)

基于 FunASR 的智能语音识别呼叫中心，作为服务器运行，处理 FreeSWITCH 发起的呼叫并进行实时语音识别与指令处理。

**启动命令:**
```bash
python ai_call_center.py
```

这将启动：
- ESL 服务器 (端口 8086)
- WebSocket 服务器 (端口 8087)

**功能特性:**
- 🔉 **实时语音识别**: 集成 FunASR（中文 Paraformer 模型）
- 🤖 **智能指令处理**: 识别“转接/挂断/重播”等口令并执行
- 🔒 **黑名单过滤**: 自动拒绝指定号码的呼叫
- 🎵 **欢迎音频**: 主音频 → 备用音频 → 音调，多级回退
- 🔄 **智能桥接**: 将呼叫桥接到指定目标
- 🛡️ **错误处理** 与 📊 **日志**

## ⚙️ 配置说明

### AI 呼叫中心配置

在 `ai_call_center.py` 中通过 `AICallCenterConfig` 配置：

```python
from ai_call_center import AICallCenterConfig

cfg = AICallCenterConfig()
cfg.blacklist_numbers = ['1002']
cfg.welcome_audio_path = '/path/to/welcome.wav'
cfg.fallback_audio = '/path/to/fallback.wav'
cfg.bridge_destination = 'user/1001@192.168.1.4'
cfg.server_host = '0.0.0.0'
cfg.server_port = 8086
cfg.funasr_model = 'paraformer-zh'
cfg.audio_sample_rate = 16000
cfg.recognition_timeout = 30
cfg.websocket_port = 8087
```

### FreeSWITCH 拨号计划配置

在 FreeSWITCH 的 `conf/dialplan/default.xml` 中添加以下配置：

```xml
<extension name="outbound_socket">
  <condition field="destination_number" expression="^4000$">
    <action application="socket" data="192.168.1.4:8086 async full"/>
  </condition>
</extension>
```

**配置说明:**
- `^4000$`: 匹配分机号 4000
- `192.168.1.4:8086`: Python 应用的 IP 和端口
- `async full`: 异步全双工模式

## 🎯 使用场景

### 场景 1: 呼叫中心系统
- 监听所有呼入事件
- 记录呼叫统计信息
- 实现智能路由和排队

### 场景 2: 语音网关
- 处理 SIP 呼叫
- 播放欢迎语音
- 桥接到目标用户

### 场景 3: 监控系统
- 实时监控呼叫状态
- 记录呼叫日志
- 异常告警

## 🔧 开发说明

### 项目结构
```
freeswitch-esl-python-sample/
├── inbound_app.py          # Inbound 应用 (事件监听)
├── ai_call_center.py       # AI 呼叫中心 (呼叫处理 + ASR)
├── funasr_recognizer.py    # FunASR 识别器模块
├── audio_stream_handler.py # 音频流处理模块 (FreeSWITCH / WebSocket)
├── websocket_client_test.py# WebSocket 测试客户端
├── web_recorder.html       # Web 录音页面
├── simple_http_server.py  # HTTP 服务器
├── test_modules.py         # 模块测试脚本
├── test_class_based_app.py # 基于类的应用测试脚本
├── requirements.txt    # Python 依赖
├── README.md          # 项目文档
└── .gitignore         # Git 忽略文件
```

### 语音识别相关

**语音指令**
- 转接人工: “转接/转人工/人工服务”
- 挂断电话: “挂断/结束/再见”
- 重播音频: “重播/再听一遍”

**工作流程**
1) 呼叫接入 → 2) 播放欢迎音频 → 3) 启动识别 → 4) 解析指令 → 5) 执行动作/桥接

**WebSocket 测试**
```bash
python websocket_client_test.py
```

**Web 录音测试**
```bash
# 启动 HTTP 服务器
python simple_http_server.py

# 在浏览器中访问
# http://localhost:8080
```

### 6. 大模型（Qwen）集成

**环境变量**
```bash
export QWEN_API_KEY=你的DashScope_API_KEY
export QWEN_MODEL=qwen-max  # 可选，默认 qwen-max
```

**说明**
- 识别到的文本将调用 Qwen 生成回复
- WebSocket 页面会收到 `type=llm_reply` 的消息并展示
- 服务器日志也会打印 LLM 回复

## 🐛 故障排除

### 常见问题

1. **连接失败**
   - 检查 FreeSWITCH 是否运行
   - 确认网络连接
   - 验证 IP 地址和端口

2. **音频播放问题**
   - 检查音频文件路径
   - 确认文件格式和采样率
   - 查看日志中的错误信息

3. **桥接失败**
   - 验证目标用户是否存在
   - 检查 SIP 配置
   - 确认网络连通性

4. **模型初始化失败 (ASR)**
   - 检查网络和磁盘空间
   - 确认 Python/依赖版本兼容

5. **识别结果不准确**
   - 检查音频质量与采样率
   - 需要时使用 GPU 加速或更换模型

6. **WebSocket 连接失败**
   - 检查端口占用/防火墙
   - 验证客户端连接参数

### 日志级别

应用支持不同级别的日志输出：
- `INFO`: 一般信息
- `WARNING`: 警告信息
- `ERROR`: 错误信息
- `DEBUG`: 调试信息

## 📝 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目！

## 📞 支持

如果您在使用过程中遇到问题，请：
1. 查看日志文件
2. 检查 FreeSWITCH 配置
3. 提交 Issue 描述问题
