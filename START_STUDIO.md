# 🚀 启动 LangGraph Studio

## 启动命令

```bash
# 确保在 ACE 环境中
conda activate ACE

# 进入项目目录
cd "/Users/delphia/Desktop/Columbia University/EECS E6893- Big Data Analytics/Medical-ACE"

# 启动 LangGraph Studio（使用隧道）
langgraph dev --tunnel
```

## 预期输出

你会看到类似这样的输出：

```
Ready!
- API: http://127.0.0.1:2024
- Studio: https://smith.langchain.com/studio/?baseUrl=https://xxx.trycloudflare.com
```

点击 Studio 链接即可在浏览器中打开可视化界面！

## 在 Studio 中使用

1. **选择 Agent**: `medical_assistant`
2. **输入查询**: 例如 "Analyze the image at 'image.png'. What is shown?"
3. **观察执行**: 看到 agent 调用 medical_vqa_tool 的过程
4. **查看结果**: 获取基于多模态模型的分析

## 故障排除

### 如果端口被占用

```bash
langgraph dev --tunnel --port 2025
```

### 如果需要本地访问（不用隧道）

```bash
langgraph dev
```

然后访问: `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`

### 检查 agent 是否正确配置

```bash
# 测试 agent 导入
python -c "from src.agents import medical_assistant; print('✅ Agent loaded successfully')"
```

---

**准备好了吗？开始吧！** 🎉

