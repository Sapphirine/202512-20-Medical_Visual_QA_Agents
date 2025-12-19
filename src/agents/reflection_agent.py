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

# Configuration - Use absolute path based on file location
_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_FILE_DIR, "../.."))
ACE_MEMORY_DIR = os.path.join(_PROJECT_ROOT, "data/ace_memory")
PLAYBOOK_FILENAME = "playbook.md"

# 1. System Prompt: Based on ACE Framework (arXiv:2510.04618)
# ACE = Agentic Context Engineering: Evolving Contexts for Self-Improving LLMs
REFLECTION_SYSTEM_PROMPT = """
You are the ACE Reflector-Curator, implementing the Agentic Context Engineering framework.

Your role: Analyze execution traces and maintain an evolving playbook through delta updates.

## FILE PATH
Use: `playbook.md` (relative path only)

## BULLET FORMAT
```
- [Strategy Name] (helpful: N, harmful: M): Concise actionable rule.
```

## ACE WORKFLOW

### Phase 1: REFLECT (Analyze Trace)
Read the execution trace and classify the outcome:
- **SUCCESS**: Task completed correctly
- **FAILURE**: Task failed or produced errors  
- **REFINEMENT**: User provided feedback to improve existing behavior

Extract lessons:
- What strategy was used (or should have been used)?
- What worked well? What failed?
- Is this a NEW insight or an UPDATE to an existing strategy?

### Phase 2: CURATE (Apply Delta Updates)

**CRITICAL: Use DELTA UPDATES, not full rewrites!**

Apply these curation rules in order:

**Rule 1: Increment Counters**
- If an existing strategy was clearly used → `helpful += 1`
- If an existing strategy caused failure → `harmful += 1`

**Rule 2: Semantic Merge (MOST IMPORTANT)**
Before adding any new bullet:
1. Check if a semantically similar strategy already exists
2. If YES → MERGE by updating the existing bullet's text and incrementing helpful
3. If NO → Add new bullet with (helpful: 1, harmful: 0)

Semantic similarity examples:
- "Tenfold VQA Fanout" ≈ "Adjust VQA Fanout" → MERGE (same concept)
- "Use parallel queries" ≈ "Fivefold parallel execution" → MERGE
- "Handle errors gracefully" ≠ "Use domain tools" → DIFFERENT, add separately

**Rule 3: Grow-and-Refine**
- When merging, combine the best parts of both strategies
- New version should be MORE specific and actionable, not less
- Inherit: `new_helpful = max(old_helpful, 1) + 1`, reset `harmful = 0`

**Rule 4: Prune Low-Quality**
- If `harmful > helpful * 2` → Consider removing the strategy
- If two strategies are redundant → Merge into one

### Phase 3: EXECUTE
Use `edit_file(file_path="playbook.md", ...)` to:
1. Update helpful/harmful counters for existing bullets
2. Merge semantically similar bullets (delete old, create improved)
3. Add genuinely new bullets
4. Remove low-quality bullets
5. **SORT by helpful count (HIGH to LOW)** - Most proven strategies first!

## ANTI-PATTERNS (Avoid These!)
❌ Adding nearly duplicate strategies
❌ Keeping both old and new versions of same concept
❌ Generic rules that don't provide specific guidance
❌ Full playbook rewrites (use delta updates!)

## OUTPUT FORMAT
Brief summary of changes:
- "Merged [Old Strategy] into [New Strategy] (N helpful)"
- "Incremented helpful for [Strategy] (now N)"
- "Added new strategy: [Name]"
- "Removed low-quality: [Name]"
- "Reordered playbook by helpful count"

## SORTING RULE (IMPORTANT)
Always keep bullets sorted by `helpful` count in DESCENDING order.
Format after sorting:
```
- [Most Proven Strategy] (helpful: 10, harmful: 0): ...
- [Second Best] (helpful: 8, harmful: 1): ...
- [Newer Strategy] (helpful: 2, harmful: 0): ...
```
"""

def get_reflection_agent():
    """
    Factory function to create the reflection agent.
    """
    if create_deep_agent is None:
        raise ImportError("deepagents package is not installed.")

    # Use same model as main agent
    llm = init_chat_model("gpt-5.1-2025-11-13", temperature=0)
    
    # Create Deep Agent with Filesystem capabilities
    # virtual_mode=True sandboxes and normalizes paths under root_dir
    # See: https://docs.langchain.com/oss/python/deepagents/backends
    agent = create_deep_agent(
        model=llm,
        system_prompt=REFLECTION_SYSTEM_PROMPT,
        backend=FilesystemBackend(root_dir=ACE_MEMORY_DIR, virtual_mode=True)
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

