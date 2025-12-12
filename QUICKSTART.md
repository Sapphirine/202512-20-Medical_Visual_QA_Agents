# 🚀 Medical-ACE Quick Start

## 启动 LangGraph Studio

### 1. 确保环境配置

```bash
# .env 文件需要包含
OPENAI_API_KEY=your_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=Medical-ACE
```

### 2. 启动开发服务器

```bash
# 在项目根目录
langgraph dev --tunnel
```

### 3. 访问 Studio

命令行会显示类似：
```
Studio URL: https://smith.langchain.com/studio/?baseUrl=https://xxx.trycloudflare.com
```

点击或复制链接到浏览器。

## 📋 已配置的 Agent

### `medical_assistant`

**功能**: 医学图像视觉问答
- **模型**: GPT-4o
- **工具**: medical_vqa_tool (CLIP + TinyLlama + Projector)

**示例提问**:
- "Analyze the image at 'image.png'. What type of tissue is shown?"
- "Look at 'image copy.png'. Is this normal or abnormal tissue?"
- "What organ is shown in this pathology image?"

## 🎯 工作流程

1. **启动 Studio** → 看到可视化界面
2. **选择 agent** → `medical_assistant`
3. **输入问题** → 提到图像路径
4. **观察执行** → 看到 agent 调用 tool 的过程
5. **获得答案** → 基于多模态模型的分析

## 🔍 调试特性

在 Studio 中你可以：
- 👁️ **可视化** agent 的思考过程
- 🔧 **调试** tool 调用
- 📊 **追踪** 完整执行链
- 🧪 **测试** 不同的 prompt
- 📈 **监控** 性能和成本

## 📂 项目结构

```
langgraph.json          # 配置文件（指向 agent）
src/agents/
  medical_assistant_agent.py   # agent 定义（模块级 agent 实例）
src/tools/
  medical_vqa_tool.py          # VQA 工具
inference.py                    # 推理引擎
projector_epoch2.pt            # 训练的模型
```

## ⚡ 本地测试（不用 Studio）

```python
from src.agents import medical_assistant

result = medical_assistant.invoke({
    "messages": [{"role": "user", "content": "Analyze image.png"}]
})

print(result["messages"][-1].content)
```

或使用测试脚本：
```bash
python test_agent.py
```

## 🐛 故障排除

### 问题：langgraph dev 找不到

```bash
pip install --upgrade langgraph-cli[inmem]
```

### 问题：Agent 创建失败

检查 `.env` 文件中的 `OPENAI_API_KEY`

### 问题：Tool 执行失败

确保 `projector_epoch2.pt` 在项目根目录

## 🎓 学习资源

- **LangGraph Studio Docs**: https://docs.langchain.com/langgraph/studio
- **create_agent API**: https://docs.langchain.com/docs/agents
- **LangSmith Tracing**: https://docs.smith.langchain.com

---

**准备好了吗？运行 `langgraph dev --tunnel` 开始！** 🚀

