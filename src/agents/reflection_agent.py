import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Load environment variables from .env file
# This will load OPENAI_API_KEY and other env vars
load_dotenv()

# Try importing deepagents, handle if missing (for valid linting in some envs)
try:
    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend
except ImportError:
    # This block is just to prevent ImportErrors if dependencies aren't installed yet
    # In production, these packages are required.
    create_deep_agent = None
    FilesystemBackend = None

# Configuration
ACE_MEMORY_DIR = os.path.abspath("./data/ace_memory")
PLAYBOOK_FILENAME = "playbook.md"

# 1. System Prompt: Focused on Reflection & Curation
REFLECTION_SYSTEM_PROMPT = """
You are the ACE (Agentic Context Engineering) Reflector and Curator.
Your goal is to maintain a high-quality 'playbook.md' by learning from execution traces.

### CORE CONCEPTS:
- **Playbook**: A list of strategic bullets.
- **Bullet Format**: `- [Strategy Name] (helpful: N, harmful: M): Detailed rule.`
- **Helpful/Harmful**: Counters tracking how often a rule helped or caused failure.

### WORKFLOW:
1. **Read** the existing 'playbook.md'.
2. **Analyze** the Execution Trace:
   - Did the agent succeed? -> **Success**
   - Did it fail or error? -> **Failure**
   - Which existing strategies were likely used (or violated)?
   
3. **Curate & Merge** (Update Logic):
   - **If Success**:
     - Identify used strategies and INCREMENT their `helpful` count.
     - If a NEW successful pattern is found, ADD it (helpful: 1, harmful: 0).
   - **If Failure**:
     - Identify responsible strategies (if any) and INCREMENT their `harmful` count.
     - Create a NEW strategy to prevent this failure (helpful: 0, harmful: 0).
   - **Refine**: 
     - If a strategy's `harmful` count is high (> `helpful`), consider REMOVING or REWRITING it.
     - Deduplicate: Do not add rules that already exist.

4. **Execute Updates**:
   - Use `edit_file` to apply changes.
   - Maintain the strict format: `- [Name] (helpful: N, harmful: M): Content`

### OUTPUT:
- "Updated playbook: incremented stats for X rules, added Y new rules."
- Or "No updates needed."
"""

def get_reflection_agent():
    """
    Factory function to create the reflection agent.
    """
    if create_deep_agent is None:
        raise ImportError("deepagents package is not installed.")

    # Use init_chat_model for flexible model initialization
    llm = init_chat_model("gpt-4.1", temperature=0)
    
    # Create Deep Agent with Filesystem capabilities
    # Note: create_deep_agent automatically attaches FilesystemMiddleware
    # We just need to provide the custom backend for the ACE memory folder
    agent = create_deep_agent(
        model=llm,
        system_prompt=REFLECTION_SYSTEM_PROMPT,
        backend=FilesystemBackend(root_dir=ACE_MEMORY_DIR)
    )
    return agent

async def process_trace_background(trace_data: str):
    """
    The interface for background processing.
    This is designed to be called by Webhooks or Background Tasks.
    """
    agent = get_reflection_agent()
    
    # Invoke the agent with the trace
    # It will use tools (read_file, edit_file) to update the playbook self-sufficiently
    response = await agent.ainvoke({
        "messages": [{"role": "user", "content": f"Analyze this trace:\n{trace_data}"}]
    })
    
    return response["messages"][-1].content

# if __name__ == "__main__":
#     # Simple verification test
#     async def test():
#         # print(f"--- ACE Memory Dir: {ACE_MEMORY_DIR} ---")
        
#         # Ensure init file exists
#         if not os.path.exists(os.path.join(ACE_MEMORY_DIR, PLAYBOOK_FILENAME)):
#             with open(os.path.join(ACE_MEMORY_DIR, PLAYBOOK_FILENAME), "w") as f:
#                 f.write("# ACE Strategy Playbook\n\n## General Strategies\n")

#         print("--- Starting Reflection Agent Test ---")
#         # Simulate a trace where the agent failed to handle a specific error
#         # mock_trace = """
#         # User: What is the result of 100 / 0?
#         # Agent: Executing division...
#         # Error: ZeroDivisionError
#         # Outcome: Failed
#         # """
#         mock_trace = """
#         User: What is DeepAgent in LangChain?
#         Tool calls: search_web
#         Agent: DeepAgent is a library for constructing advanced agent systems.
#         User: There is an mcp tool for langchain doc, use it to answer related questions.
#         """
        
#         mock_trace = """
#         User: What is DeepAgent in LangChain?
#         Tool used: mcp_Docs_by_LangChain_SearchDocsByLangChain
#         Agent: Based on the official LangChain documentation...
#         User: Thank you for the information.
#         """

#         mock_trace = """
#         User: What are the new features in LangGraph 0.2?
#         Tool used: mcp_LegacyLangChain_Docs_Search
#         Tool Response: No results found. This documentation source is deprecated and no longer maintained.
#         Agent: I apologize, but the documentation tool I used is deprecated and returned no information.
#         User: Please remove the deprecated mcp_LegacyLangChain_Docs_Search tool from your toolkit and use mcp_Docs_by_LangChain_SearchDocsByLangChain instead.
#         Tool used: remove_tool
#         Tool Input: {"tool_name": "mcp_LegacyLangChain_Docs_Search"}
#         Agent: I've removed the deprecated tool. Let me try again with the updated documentation source.
#         """
        
#         # mock_trace = """
#         # User: How do Agno orchestrate agents?
#         # Agent: there is no mcp tool for Agno.
#         # Tool used: search_web
#         # Agent: Agno is a platform for orchestrating agents. It provides a set of tools for agents to communicate and collaborate.
#         # """
        
#         try:
#             result = await process_trace_background(mock_trace)
#             print(f"Agent Response: {result}")
            
#             # Verify file content
#             with open(os.path.join(ACE_MEMORY_DIR, PLAYBOOK_FILENAME), "r") as f:
#                 print("\n--- Playbook Content After Update ---")
#                 print(f.read())
#         except Exception as e:
#             print(f"Test failed (likely due to missing dependencies or API key): {e}")

#     asyncio.run(test())

