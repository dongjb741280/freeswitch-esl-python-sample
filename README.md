
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
   git clone <repository-url>
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

### Outbound 应用 (呼叫处理)

Outbound 应用作为服务器运行，处理 FreeSWITCH 发起的呼叫，实现智能呼叫路由。

**启动命令:**
```bash
python outbound_app.py
```

**功能特性:**
- 🔒 **黑名单过滤**: 自动拒绝指定号码的呼叫
- 🎵 **欢迎音频**: 为合法呼叫播放背景音乐
- 🔄 **智能桥接**: 将呼叫桥接到指定目标
- 🛡️ **错误处理**: 完善的异常处理和恢复机制
- 📊 **详细日志**: 记录所有操作和状态变化

## ⚙️ 配置说明

### Outbound 应用配置

在 `outbound_app.py` 中的 `CONFIG` 字典可以配置以下参数：

```python
CONFIG = {
    'blacklist_numbers': ['1002'],  # 黑名单号码
    'welcome_audio_path': '/path/to/welcome.wav',  # 主音频文件
    'fallback_audio': '/path/to/fallback.wav',     # 备用音频文件
    'bridge_destination': 'user/1001@192.168.1.4', # 桥接目标
    'server_host': '0.0.0.0',                       # 监听地址
    'server_port': 8086                           # 监听端口
}
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
├── inbound_app.py      # Inbound 应用 (事件监听)
├── outbound_app.py     # Outbound 应用 (呼叫处理)
├── requirements.txt    # Python 依赖
├── README.md          # 项目文档
└── .gitignore         # Git 忽略文件
```

### 核心功能模块

**Outbound 应用主要函数:**
- `establish_connection()` - 建立 ESL 连接
- `answer_call()` - 应答呼叫
- `get_caller_info()` - 获取主叫信息
- `play_welcome_audio()` - 播放欢迎音频
- `bridge_call()` - 桥接呼叫

**音频处理策略:**
1. 主音频文件 (48kHz 高质量)
2. 备用音频文件 (8kHz 标准)
3. 音调生成 (800Hz 持续音)
4. 错误提示音 (480Hz+620Hz 双音)

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
