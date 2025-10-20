# 解耦ReAct智能体

基于LangGraph + 异步中断恢复理论实现的工具调用与用户交互分离的智能体系统。

## 🎯 核心特性

### 1. 工具与交互解耦
- **异步工具执行**: 工具调用不阻塞用户交互
- **并行处理**: 多个工具可同时执行
- **实时通知**: 工具状态变化实时反馈

### 2. 流式输出中断与恢复
- **可中断流式输出**: 基于异步机制实现流式输出中断
- **智能恢复**: 工具完成后自动恢复并整合结果
- **上下文保持**: 中断恢复过程中保持完整对话上下文

### 3. ReAct模式增强
- **思考-行动-观察**: 经典ReAct模式
- **异步行动**: 行动(工具调用)异步执行
- **持续思考**: 工具执行期间可继续与用户交流

## 🏗️ 系统架构

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   用户界面      │    │  解耦管理器      │    │   ReAct智能体   │
│  DemoInterface  │◄──►│DecoupledManager │◄──►│  ReactAgent     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │                         │
                              ▼                         ▼
                    ┌──────────────────┐    ┌─────────────────┐
                    │  中断管理器      │    │   工具注册器    │
                    │InterruptManager │    │  ToolRegistry   │
                    └──────────────────┘    └─────────────────┘
```

### 核心组件

1. **DecoupledReActManager**: 解耦管理器
   - 管理工具调用与用户交互的分离
   - 处理异步任务调度
   - 协调流式输出中断与恢复

2. **ReactAgent**: ReAct智能体
   - 实现ReAct推理模式
   - 解析工具调用指令
   - 生成流式响应

3. **AsyncInterruptManager**: 异步中断管理器
   - 管理流式输出会话
   - 实现中断与恢复机制
   - 维护会话状态

4. **ToolRegistry**: 工具注册器
   - 管理可用工具
   - 支持同步/异步工具
   - 提供工具执行接口

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境

复制 `.env` 文件并配置你的API密钥：

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL_NAME=gpt-3.5-turbo
```

### 3. 运行演示

```bash
python main.py
```

## 💡 使用示例

### 基础对话
```
> 你好，请帮我计算 123 + 456

助手: 我来帮你计算这个数学表达式。

思考: 用户需要计算 123 + 456，我需要使用计算器工具。
行动: 计算器
行动输入: {"expression": "123 + 456"}

🔧 正在执行 1 个工具，我们可以继续对话...

同时，如果你还有其他问题，我们可以继续交流。

✅ 工具执行完成: 计算器 (耗时: 0.01s)
   结果: 计算结果：123 + 456 = 579

⚡ 流式输出被中断，工具 计算器 已完成

基于计算结果，123 + 456 = 579。
```

### 复杂任务（多工具并行）
```
> 请帮我查询北京的天气，同时搜索一下人工智能的最新发展

助手: 我来帮你同时查询天气和搜索信息。

思考: 用户需要两个不同的信息，我可以同时执行两个工具。
行动: 天气查询
行动输入: {"city": "北京"}
行动: 搜索
行动输入: {"query": "人工智能最新发展"}

🔧 正在执行 2 个工具，我们可以继续对话...

这两个查询会并行执行，预计几秒钟内完成。在等待期间，你还有什么想了解的吗？

✅ 工具执行完成: 天气查询 (耗时: 3.02s)
✅ 工具执行完成: 搜索 (耗时: 2.15s)

⚡ 流式输出被中断，工具 搜索 已完成

根据查询结果：

**北京天气**: 晴天，温度25°C，湿度60%

**人工智能最新发展**: 
- 大语言模型技术持续突破...
- 多模态AI应用日益成熟...
- AI安全和伦理问题受到重视...
```

## 🔧 可用工具

- **搜索**: 模拟网络搜索功能
- **计算器**: 数学表达式计算
- **获取时间**: 获取当前时间
- **文件读取**: 文件内容读取
- **天气查询**: 城市天气查询
- **翻译**: 文本翻译功能

## 📋 命令列表

- `/status` - 查看任务执行状态
- `/history` - 查看对话历史
- `/reset` - 重置对话
- `/quit` - 退出程序

## 🎨 技术亮点

### 1. 异步架构设计
```python
# 工具异步执行，不阻塞用户交互
async def _execute_tool_async(self, task: ToolTask):
    task.status = TaskStatus.RUNNING
    result = await self.agent.execute_tool(task.tool_name, task.tool_input)
    # 完成后自动中断并恢复流式输出
    await self._interrupt_and_provide_tool_result(task)
```

### 2. 流式输出中断机制
```python
# 检测中断并恢复
if session and session.status == StreamStatus.INTERRUPTED:
    break
# 恢复时整合工具结果
await self._resume_with_tool_result(tool_result_message)
```

### 3. 智能上下文管理
```python
# 保持完整对话历史
self.conversation_history.append(ToolMessage(
    content=completed_task.result,
    tool_call_id=completed_task.id
))
```

## 🔬 理论基础

### LLM流式输出特性
- LLM流式输出是边推理边输出，为中断奠定基础
- 异步机制可以中断流式输出连接
- 模型端也会停止输出并标记为完成

### 恢复机制原理
- LLM是自回归模型，根据上下文可以继续推理
- 中断恢复本质是存储中断前信息并重新发送
- 可以通过空字符串"继续"等提示使模型继续输出

### API格式要求
- 必须以user/system消息开头
- 最后的消息可以是任意类型
- 相同类型消息不可连续出现
- LLM本身对数据类型不敏感，限制来自接口层

## 🛠️ 扩展开发

### 添加新工具

1. 在 `tool/tool_registry.py` 中注册新工具：

```python
@register_tool("新工具名称")
async def new_tool(param1: str, param2: int) -> str:
    """新工具描述"""
    # 工具实现逻辑
    await asyncio.sleep(1)  # 模拟异步操作
    return "工具执行结果"
```

2. 工具会自动被智能体识别和使用

### 自定义回调

```python
manager = DecoupledReActManager()

# 设置自定义回调
manager.on_tool_start = custom_tool_start_handler
manager.on_tool_complete = custom_tool_complete_handler
manager.on_stream_interrupt = custom_interrupt_handler
```

## 📈 性能优化

- **并行工具执行**: 多个工具同时运行，减少总等待时间
- **流式输出**: 实时响应，提升用户体验
- **智能中断**: 仅在必要时中断，避免频繁打断
- **上下文复用**: 高效的消息历史管理

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

MIT License

## 🙋‍♂️ 常见问题

**Q: 工具执行失败怎么办？**
A: 系统会自动捕获错误并在任务状态中显示，可通过 `/status` 命令查看详情。

**Q: 如何添加更多工具？**
A: 在 `tool_registry.py` 中使用 `@register_tool` 装饰器注册新工具即可。

**Q: 支持哪些LLM模型？**
A: 支持所有兼容OpenAI API格式的模型，包括GPT系列、Claude等。

**Q: 如何处理长时间运行的工具？**
A: 系统支持异步执行，长时间工具不会阻塞用户交互，完成后会自动通知。