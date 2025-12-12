"""
Medical Assistant Agent

An intelligent agent that can answer questions about medical images
using the multimodal VQA tool.

This agent is automatically exposed by langgraph dev.
"""

import os
import sys
from dotenv import load_dotenv
from langchain.agents import create_agent

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

Tool call examples:

For uploaded image (CLEAN & FAST):
  image_input: "UPLOADED_IMAGE"
  question: "What does this show?"

For existing file:
  image_input: "image.png"
  question: "What tissue is shown?"

Be concise but thorough in your responses.
"""


# Create image handler middleware to process uploaded images
image_middleware = ImageHandlerMiddleware(save_dir=project_root)

# Create the agent directly at module level
# langgraph dev will automatically discover this
agent = create_agent(
    model="openai:gpt-4o",  # Using gpt-4o as default
    tools=[medical_vqa_tool],
    system_prompt=SYSTEM_PROMPT,
    middleware=[image_middleware],  # Add image handling middleware
)


# Optional: For direct testing
if __name__ == "__main__":
    print("=== Medical Assistant Agent ===\n")
    
    # Test query
    test_query = "Please analyze the image at 'image.png' and tell me what type of tissue is shown."
    
    print(f"Query: {test_query}\n")
    print("Processing...\n")
    
    try:
        result = agent.invoke({
            "messages": [{"role": "user", "content": test_query}]
        })
        
        response = result["messages"][-1].content
        print(f"✅ Response:\n{response}\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nNote: Make sure OPENAI_API_KEY is set in your .env file")
