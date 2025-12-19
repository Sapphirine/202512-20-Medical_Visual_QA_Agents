# Medical-ACE: Medical Visual Question Answering System

🏥 AI-powered medical image analysis system combining vision and language models.

*EECS6893 Big Data Analytics Final Project: A Multi-Agent System that Reads and Understands Medical Images at Scale*

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Activate conda environment
conda activate ACE

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Create `.env` file:

```bash
OPENAI_API_KEY=your_openai_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=Medical-ACE
```

### 3. Run the System

#### Option A: LangGraph Studio (Recommended)

```bash
langgraph dev --tunnel
```

Then visit the URL shown in terminal to use the visual Studio interface.

#### Option B: Command Line

```bash
# Single query
python main.py --mode single

# Batch processing
python main.py --mode batch

# Interactive mode
python main.py --mode interactive
```

#### Option C: Test Script

```bash
python test_agent.py
```

## 🏗️ Architecture

```
Medical-ACE/
├── src/
│   ├── agents/
│   │   └── medical_assistant_agent.py  # Main agent (GPT-4o + VQA tool)
│   └── tools/
│       └── medical_vqa_tool.py         # Medical VQA tool
├── inference.py                         # Multimodal inference engine
├── projector_epoch2.pt                 # Trained projector model
├── main.py                             # CLI entry point
└── langgraph.json                      # LangGraph configuration
```

## 🔧 Components

### Agent
- **Model**: GPT-4o (OpenAI)
- **Type**: LangChain `create_agent`
- **Tools**: Medical VQA Tool

### Medical VQA Tool
- **Vision**: CLIP (openai/clip-vit-base-patch32)
- **Language**: TinyLlama (1.1B)
- **Projector**: Custom trained (epoch 2)

## 📊 Supported Image Types

- 🔬 Pathology slides
- 🩻 X-rays
- 🧠 CT/MRI scans
- 🫀 Other medical imaging

## 💡 Usage Examples

### Python API

```python
from src.agents import medical_assistant

# Query the agent
result = medical_assistant.invoke({
    "messages": [{"role": "user", "content": "Analyze image.png"}]
})

print(result["messages"][-1].content)
```

### LangGraph Studio

1. Start server: `langgraph dev --tunnel`
2. Open Studio in browser
3. Select `medical_assistant` agent
4. Chat with the agent about medical images

## 🔬 Testing

```bash
# Test the agent
python test_agent.py

# Test inference module directly
python inference.py image.png "What is shown?" projector_epoch2.pt
```

## ⚙️ Configuration

Edit `src/agents/medical_assistant_agent.py` to customize:
- Model selection
- System prompt
- Tools

Edit `langgraph.json` to add more agents.

## 📝 Notes

⚠️ **Medical Disclaimer**: This system is for research/education only. AI analysis should be verified by qualified medical professionals.

## 🔗 Resources

- [LangChain Docs](https://docs.langchain.com)
- [LangGraph Studio](https://docs.langchain.com/langgraph/studio)
- [Project Documentation](docs/MEDICAL_ASSISTANT_GUIDE.md)

## 📄 License

MIT License
