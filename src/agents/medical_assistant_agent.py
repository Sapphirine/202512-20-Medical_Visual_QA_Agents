"""
Medical Assistant Agent

An intelligent agent that can answer questions about medical images
using the multimodal VQA tool.

This agent is automatically exposed by langgraph dev.
"""

import os
import sys
from dotenv import load_dotenv

from typing import Annotated, TypedDict, Union, List, Dict, Any
import operator

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

# Load environment variables
load_dotenv()

# Setup path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.tools.medical_vqa_tool import medical_vqa_tool
from src.middleware.image_handler import ImageHandlerMiddleware


# System prompt for the medical assistant
SYSTEM_PROMPT = """You are a helpful medical imaging assistant powered by AI.

You have access to a powerful medical visual question answering (VQA) tool that can analyze:
- Pathology slides (histopathology images)
- X-rays
- CT scans
- MRI images
- Other medical imaging modalities

When a user asks about a medical image:
1. Use the medical_vqa_tool to analyze the image
2. Provide clear, informative responses based on the tool's analysis
3. If needed, ask follow-up questions for clarification
4. Always remind users that AI analysis should be verified by qualified medical professionals

CRITICAL: HOW TO HANDLE IMAGES:

1. IF USER UPLOADS AN IMAGE:
   - DO NOT extract the base64 string (it's too long and messy)
   - JUST use the magic string "UPLOADED_IMAGE" as the image_input
   - The system middleware will automatically handle the image data in the background
   - Example: image_input="UPLOADED_IMAGE"

2. IF USER MENTIONS A FILE PATH:
   - Use the file path directly (e.g., "image.png")

3. NEVER make up file paths.
4. DO NOT reuse 'uploaded_*.png' paths from previous conversation turns. ALWAYS use "UPLOADED_IMAGE" if you want to analyze the currently uploaded image.

Tool call examples:

For uploaded image (CLEAN & FAST):
  image_input: "UPLOADED_IMAGE"
  question: "What does this show?"

For existing file:
  image_input: "image.png"
  question: "What tissue is shown?"

Be concise but thorough in your responses.
"""

# Define State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    _uploaded_image_path: Union[str, None]

# 1. State Processing (Middleware logic)
image_middleware = ImageHandlerMiddleware(save_dir=project_root)

def preprocess_state(state: AgentState) -> Dict[str, Any]:
    """Runs image middleware logic to save uploaded images."""
    # This middleware modifies state in-place conceptually or returns new state
    # ImageHandlerMiddleware.before_agent returns the full state dict
    
    # We pass a copy to avoid side-effects issues if any
    state_copy = state.copy()
    
    # The middleware expects a dict with "messages"
    updated_state = image_middleware.before_agent(state_copy)
    
    # Extract the special key if it was set
    return {
        "_uploaded_image_path": updated_state.get("_uploaded_image_path")
    }

# 2. Model Node
tools = [medical_vqa_tool]
# Use GPT-4o
model = ChatOpenAI(model="gpt-4o")
model_with_tools = model.bind_tools(tools)

def call_model(state: AgentState, config: RunnableConfig):
    messages = state["messages"]
    # Prepend system prompt for the context
    # We don't add it to history, just for this call
    system_msg = SystemMessage(content=SYSTEM_PROMPT)
    chain_input = [system_msg] + messages
    
    response = model_with_tools.invoke(chain_input, config)
    return {"messages": [response]}

# 3. Tool Node (Custom to inject image path)
def custom_tool_node(state: AgentState):
    """Executes tools, injecting uploaded image path if needed."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Ensure current_image_path is set in middleware based on state
    uploaded_path = state.get("_uploaded_image_path")
    image_middleware.current_image_path = uploaded_path
    
    # We need to modify the tool calls in the last message if they are using placeholders
    tool_calls = last_message.tool_calls
    
    # We modify the tool calls in place (or their copies)
    # LangChain tool_calls are dicts: {'name':..., 'args':..., 'id':...}
    
    for tool_call in tool_calls:
        # This modifies tool_call['args'] in place if needed
        image_middleware.wrap_tool_call(tool_call, state)
        
    # Now execute tools using standard ToolNode
    # We construct a new ToolNode on the fly or reuse one
    tool_node = ToolNode(tools)
    
    return tool_node.invoke(state)


# Build Graph
builder = StateGraph(AgentState)

builder.add_node("preprocess", preprocess_state)
builder.add_node("agent", call_model)
builder.add_node("tools", custom_tool_node)

builder.set_entry_point("preprocess")

builder.add_edge("preprocess", "agent")

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

builder.add_conditional_edges("agent", should_continue)
builder.add_edge("tools", "agent")

# Compile
agent = builder.compile()

# Optional: For direct testing
if __name__ == "__main__":
    print("=== Medical Assistant Agent (LangGraph) ===\n")
    
    test_query = "Please analyze the image at 'image.png' and tell me what type of tissue is shown."
    
    try:
        inputs = {"messages": [("user", test_query)]}
        for output in agent.stream(inputs):
            for key, value in output.items():
                print(f"Output from node '{key}':")
                print("---")
                if key == "agent":
                    print(value["messages"][-1].content)
                else:
                    print(value)
                print("\n---\n")
                
    except Exception as e:
        print(f"❌ Error: {e}")
