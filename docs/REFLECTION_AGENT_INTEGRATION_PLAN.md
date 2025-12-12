# Reflection Agent 集成计划

> **目标**: 将 `reflection_agent.py` 集成到 `medical_assistant_agent.py` 主 Agent 中，实现 ACE (Agentic Context Engineering) 的自我改进闭环。

---

## 🆕 Playbook Prompt 动态注入方案

> **用户问题**: 如何将 `data/ace_memory/playbook.md` 的内容动态注入到 medical agent 的 system prompt 中？

根据 LangChain 官方文档，有以下三种推荐方案：

### 方案对比

| 方案 | 复杂度 | 实时性 | 适用场景 | 文档来源 |
|------|--------|--------|----------|----------|
| **A. f-string 动态读取** | ⭐ 简单 | 每次调用时读取 | 快速原型 | LangGraph quickstart |
| **B. Middleware wrap_model_call** | ⭐⭐ 中等 | 拦截每次 LLM 调用 | 生产推荐 | [context-engineering](https://docs.langchain.com/oss/python/langchain/context-engineering) |
| **C. LangSmith Store** | ⭐⭐⭐ 复杂 | 持久化存储 | 多 agent 共享 | [context-engineering](https://docs.langchain.com/oss/python/langchain/context-engineering) |

---

### 方案 A: f-string 动态读取 (推荐开始使用)

最简单直接的方式，在 `call_model` 节点中动态读取 playbook 文件：

```python
# 在 medical_assistant_agent.py 中修改

import os

# Playbook 路径
ACE_PLAYBOOK_PATH = os.path.join(
    os.path.dirname(__file__), 
    "../../data/ace_memory/playbook.md"
)

def get_playbook_content() -> str:
    """动态读取 playbook 内容"""
    try:
        with open(ACE_PLAYBOOK_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "# No playbook available yet."

# 修改 SYSTEM_PROMPT 为动态函数
def get_system_prompt() -> str:
    playbook = get_playbook_content()
    return f"""You are a helpful medical imaging assistant powered by AI.

You have access to a powerful medical visual question answering (VQA) tool...

[原有的 SYSTEM_PROMPT 内容...]

---

## ACE Strategy Playbook (动态更新)

以下是从执行经验中学到的策略，请参考这些策略做出更好的决策：

{playbook}

---
"""

# 修改 call_model 函数
def call_model(state: AgentState, config: RunnableConfig):
    messages = state["messages"]
    # 每次调用时动态获取最新的 system prompt
    system_msg = SystemMessage(content=get_system_prompt())
    chain_input = [system_msg] + messages
    
    response = model_with_tools.invoke(chain_input, config)
    return {"messages": [response]}
```

**优点**: 
- 实现简单，无需额外依赖
- 每次 LLM 调用都获取最新的 playbook
- 与现有代码结构兼容

**缺点**:
- 文件 I/O 在每次调用时发生
- 没有缓存机制

---

### 方案 B: Middleware wrap_model_call (生产推荐)

使用 LangChain v1 的 middleware 模式，更优雅地注入上下文：

```python
# 文档来源: https://docs.langchain.com/oss/python/langchain/context-engineering

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable

ACE_PLAYBOOK_PATH = "./data/ace_memory/playbook.md"

@wrap_model_call
def inject_playbook_context(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """
    拦截每次 LLM 调用，注入 playbook 上下文。
    
    文档参考: https://docs.langchain.com/oss/python/langchain/context-engineering
    """
    # 读取 playbook
    try:
        with open(ACE_PLAYBOOK_PATH, "r") as f:
            playbook_content = f.read()
    except FileNotFoundError:
        playbook_content = "No strategies available."
    
    # 构建注入消息
    playbook_message = {
        "role": "system",
        "content": f"""
## ACE Strategy Playbook

The following strategies were learned from past executions. 
Use them to guide your decisions:

{playbook_content}
"""
    }
    
    # 将 playbook 添加到消息末尾 (LLM 更关注末尾内容)
    messages = [*request.messages, playbook_message]
    request = request.override(messages=messages)
    
    return handler(request)

# 使用 middleware 创建 agent
agent = create_agent(
    model="gpt-4o",
    tools=[medical_vqa_tool],
    middleware=[inject_playbook_context]  # 添加 middleware
)
```

**优点**:
- 遵循 LangChain 最佳实践
- 关注点分离：prompt 逻辑与业务逻辑解耦
- 可组合多个 middleware

**缺点**:
- 需要 LangChain v1 / langchain-agents 库
- 语法与当前 LangGraph StateGraph 模式不同

---

### 方案 C: LangSmith Store (高级)

使用 LangGraph Store 持久化 playbook，支持跨 session 和多 agent 共享：

```python
# 文档来源: https://docs.langchain.com/oss/python/langchain/context-engineering

from langgraph.store.memory import InMemoryStore
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable

@wrap_model_call
def inject_playbook_from_store(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """
    从 LangGraph Store 读取 playbook。
    """
    store = request.runtime.store  # 访问 Store
    
    # 从 Store 获取 playbook
    playbook_item = store.get(("ace",), "playbook")
    
    if playbook_item:
        playbook_content = playbook_item.value.get("content", "")
        
        # 注入到消息中
        messages = [
            *request.messages,
            {"role": "system", "content": f"## Strategy Playbook\n{playbook_content}"}
        ]
        request = request.override(messages=messages)
    
    return handler(request)

# 创建 Store 并初始化 playbook
store = InMemoryStore()

# 读取文件并存入 Store
with open("./data/ace_memory/playbook.md", "r") as f:
    store.put(("ace",), "playbook", {"content": f.read()})

# 创建 agent
agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[inject_playbook_from_store],
    store=store
)
```

**优点**:
- 支持语义搜索 (如果启用 index)
- 可跨 thread/session 共享
- 反思 agent 可以直接更新 Store

**缺点**:
- 需要额外的 Store 基础设施
- 引入状态同步复杂性

---

### 🎯 推荐实现路径

1. **Phase 1 (现在)**: 使用 **方案 A** - f-string 动态读取
   - 最快实现，立即可用
   - 与现有 LangGraph StateGraph 完全兼容

2. **Phase 2 (优化)**: 添加缓存机制
   ```python
   import functools
   import time
   
   @functools.lru_cache(maxsize=1)
   def get_playbook_cached(mtime: float) -> str:
       with open(ACE_PLAYBOOK_PATH, "r") as f:
           return f.read()
   
   def get_playbook_content() -> str:
       mtime = os.path.getmtime(ACE_PLAYBOOK_PATH)
       return get_playbook_cached(mtime)
   ```

3. **Phase 3 (生产)**: 迁移到 **方案 B** middleware 模式

---

## 1. 现状分析

### 1.1 主 Agent (`medical_assistant_agent.py`)
- **框架**: LangGraph StateGraph
- **State 结构**:
  ```python
  class AgentState(TypedDict):
      messages: Annotated[List[BaseMessage], add_messages]
      _uploaded_image_path: Union[str, None]
  ```
- **节点**: `preprocess` → `agent` → `tools` (循环)
- **部署**: 通过 `langgraph.json` 暴露为 `medical_assistant`

### 1.2 Reflection Agent (`reflection_agent.py`)
- **框架**: `deepagents` 库 (基于 LangGraph)
- **功能**: 分析执行 trace，更新 `data/ace_memory/playbook.md`
- **接口**: `process_trace_background(trace_data: str)` - 异步函数

---

## 2. 集成方案对比 (基于 LangChain 官方文档)

根据 LangGraph 文档，有三种主要集成方式：

| 方案 | 描述 | 优点 | 缺点 | 文档来源 |
|------|------|------|------|----------|
| **A. Subgraph 节点调用** | 在父图节点内调用子图 | 完全控制 state 转换 | 同步执行，会阻塞主流程 | [use-subgraphs](https://docs.langchain.com/oss/python/langgraph/use-subgraphs) |
| **B. asyncio.create_task** | Fire-and-forget 后台任务 | 不阻塞主流程 | 不与 LangGraph state 直接绑定 | Python asyncio 标准模式 |
| **C. LangSmith Background Run** | 通过 webhook 触发后台运行 | 生产级、可监控 | 需要 LangSmith 部署 | [create-background-run](https://docs.langchain.com/langsmith/agent-server-api/thread-runs/create-background-run) |

### 推荐方案: **方案 B + C 组合**
- **开发阶段**: 使用 `asyncio.create_task` 实现本地 fire-and-forget
- **生产阶段**: 迁移到 LangSmith Background Runs + Webhooks

---

## 3. 详细实现计划

### 3.1 Phase 1: 状态扩展

**目标**: 扩展 `AgentState` 以支持 trace 收集

```python
# 在 medical_assistant_agent.py 中修改

from typing import Annotated, TypedDict, Union, List, Dict, Any, Optional
import operator

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    _uploaded_image_path: Union[str, None]
    # 新增: 用于 ACE 反思的 trace 数据
    _execution_trace: Optional[str]  # 收集的执行轨迹
    _reflection_triggered: bool  # 是否已触发反思
```

### 3.2 Phase 2: Trace 收集节点

**目标**: 在 agent 执行结束时收集 trace

```python
# 新增节点函数

def collect_trace(state: AgentState) -> Dict[str, Any]:
    """
    收集执行轨迹用于反思。
    在图的 END 之前运行。
    """
    messages = state["messages"]
    
    # 构建 trace 字符串
    trace_parts = []
    for msg in messages:
        role = getattr(msg, 'type', 'unknown')
        content = getattr(msg, 'content', '')[:500]  # 截断过长内容
        
        # 检查是否有 tool calls
        tool_calls = getattr(msg, 'tool_calls', [])
        if tool_calls:
            tool_info = ", ".join([tc.get('name', 'unknown') for tc in tool_calls])
            trace_parts.append(f"{role}: [Tool Calls: {tool_info}] {content}")
        else:
            trace_parts.append(f"{role}: {content}")
    
    execution_trace = "\n".join(trace_parts)
    
    return {
        "_execution_trace": execution_trace,
        "_reflection_triggered": False  # 标记尚未触发
    }
```

### 3.3 Phase 3: Fire-and-Forget 反思触发

**目标**: 后台触发 reflection agent，不阻塞主流程

#### 方式 A: 使用 asyncio (开发阶段推荐)

```python
import asyncio
from src.agents.reflection_agent import process_trace_background

# 全局任务追踪器 (可选，用于监控)
_background_tasks = set()

async def trigger_reflection_async(trace_data: str):
    """
    Fire-and-forget 后台反思任务。
    """
    try:
        result = await process_trace_background(trace_data)
        print(f"[ACE Reflection] Completed: {result[:100]}...")
    except Exception as e:
        print(f"[ACE Reflection] Error: {e}")
    finally:
        # 清理任务引用
        pass

def trigger_reflection_node(state: AgentState) -> Dict[str, Any]:
    """
    触发后台反思的节点。
    使用 asyncio.create_task 实现 fire-and-forget。
    """
    trace = state.get("_execution_trace", "")
    
    if not trace or state.get("_reflection_triggered", False):
        return {"_reflection_triggered": True}
    
    # 获取当前事件循环
    try:
        loop = asyncio.get_running_loop()
        # 创建后台任务 (fire-and-forget)
        task = loop.create_task(trigger_reflection_async(trace))
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
    except RuntimeError:
        # 如果没有运行中的事件循环，使用线程池
        import concurrent.futures
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        executor.submit(asyncio.run, trigger_reflection_async(trace))
    
    return {"_reflection_triggered": True}
```

#### 方式 B: 使用 Webhook (生产阶段)

```python
# 参考 LangSmith 文档:
# https://docs.langchain.com/langsmith/use-webhooks

import httpx

async def trigger_reflection_via_webhook(trace_data: str):
    """
    通过 webhook 触发 LangSmith 后台运行。
    """
    webhook_url = "https://your-deployment/api/reflection/trigger"
    
    async with httpx.AsyncClient() as client:
        await client.post(
            webhook_url,
            json={"trace": trace_data},
            timeout=5.0  # 快速返回，不等待完成
        )
```

### 3.4 Phase 4: 图结构更新

**目标**: 将新节点集成到图中

```python
# 更新后的图结构

from langgraph.graph import StateGraph, END, START

builder = StateGraph(AgentState)

# 现有节点
builder.add_node("preprocess", preprocess_state)
builder.add_node("agent", call_model)
builder.add_node("tools", custom_tool_node)

# 新增节点
builder.add_node("collect_trace", collect_trace)
builder.add_node("trigger_reflection", trigger_reflection_node)

# 边定义
builder.set_entry_point("preprocess")
builder.add_edge("preprocess", "agent")

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return "collect_trace"  # 修改: 不直接 END

builder.add_conditional_edges("agent", should_continue)
builder.add_edge("tools", "agent")

# 新增: 反思流程
builder.add_edge("collect_trace", "trigger_reflection")
builder.add_edge("trigger_reflection", END)

agent = builder.compile()
```

### 3.5 图结构可视化

```
                    ┌─────────────────┐
                    │   preprocess    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
            ┌──────▶│     agent       │◀─────┐
            │       └────────┬────────┘      │
            │                │               │
            │       has_tool_calls?          │
            │         /          \           │
            │       yes           no         │
            │        │             │         │
            │        ▼             ▼         │
            │  ┌──────────┐  ┌─────────────────┐
            │  │  tools   │  │  collect_trace  │
            │  └────┬─────┘  └────────┬────────┘
            │       │                 │
            └───────┘                 ▼
                            ┌─────────────────────┐
                            │ trigger_reflection  │ (Fire-and-Forget)
                            └────────┬────────────┘
                                     │
                                     ▼
                                   [END]
```

---

## 4. 如何获取 State (关键文档参考)

根据 LangGraph 官方文档，在节点中获取 state 的方式：

### 4.1 节点函数参数

```python
# 文档来源: https://docs.langchain.com/oss/python/langgraph/graph-api

def my_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    节点函数的第一个参数始终是 state。
    
    Args:
        state: 当前图的完整状态
        config: 包含 thread_id、tags 等配置信息
    
    Returns:
        状态更新字典 (不是完整 state，只是增量更新)
    """
    # 读取 state
    messages = state["messages"]
    uploaded_path = state.get("_uploaded_image_path")
    
    # 返回更新 (只包含要更新的键)
    return {
        "_execution_trace": "some trace data"
    }
```

### 4.2 在 Subgraph 中传递 State

```python
# 文档来源: https://docs.langchain.com/oss/python/langgraph/use-subgraphs

def invoke_subgraph_node(state: ParentState) -> Dict[str, Any]:
    """
    在父图节点中调用子图。
    需要手动进行 state 转换。
    """
    # 转换到子图 state
    subgraph_input = {"bar": state["foo"]}
    
    # 调用子图
    subgraph_output = subgraph.invoke(subgraph_input)
    
    # 转换回父图 state
    return {"foo": subgraph_output["bar"]}
```

### 4.3 通过 Config 获取元数据

```python
# 文档来源: https://docs.langchain.com/oss/javascript/langgraph/graph-api

def my_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    # 获取当前步骤数
    current_step = config.get("metadata", {}).get("langgraph_step", 0)
    
    # 获取 thread_id
    thread_id = config.get("configurable", {}).get("thread_id")
    
    return state
```

---

## 5. 可选增强: 使用 deepagents SubAgentMiddleware

根据 deepagents 文档，可以使用 SubAgentMiddleware 实现更优雅的子 agent 集成：

```python
# 文档来源: https://docs.langchain.com/oss/python/deepagents/middleware

from deepagents.middleware.subagents import SubAgentMiddleware

# 定义 reflection 作为 subagent
reflection_subagent = {
    "name": "ace_reflector",
    "description": "Analyzes execution traces and updates the strategy playbook",
    "system_prompt": REFLECTION_SYSTEM_PROMPT,
    "tools": []  # 使用 FilesystemBackend
}

# 在主 agent 创建时添加 middleware
agent = create_deep_agent(
    model=llm,
    system_prompt=MAIN_SYSTEM_PROMPT,
    middleware=[
        SubAgentMiddleware(subagents=[reflection_subagent])
    ]
)
```

**注意**: 此方案需要将主 agent 迁移到 deepagents 框架，工作量较大。

---

## 6. 实施检查清单

- [ ] **Phase 1**: 扩展 `AgentState` TypedDict
- [ ] **Phase 2**: 实现 `collect_trace` 节点函数
- [ ] **Phase 3**: 实现 `trigger_reflection_node` 节点函数
- [ ] **Phase 4**: 更新图结构 (edges)
- [ ] **Phase 5**: 测试 fire-and-forget 行为
- [ ] **Phase 6**: 验证 playbook.md 更新
- [ ] **Phase 7**: (生产) 迁移到 LangSmith Background Runs

---

## 7. 风险与注意事项

1. **事件循环冲突**: `langgraph dev` 已有运行的事件循环，需使用 `loop.create_task()` 而非 `asyncio.run()`
2. **State 隔离**: reflection agent 不应修改主 agent 的 state
3. **错误处理**: 后台任务的异常不应影响主流程
4. **资源泄漏**: 使用 `task.add_done_callback()` 清理任务引用
5. **trace 大小**: 截断过长的 message content 避免 token 溢出

---

## 8. 参考文档链接

| 主题 | 链接 |
|------|------|
| LangGraph Subgraphs | https://docs.langchain.com/oss/python/langgraph/use-subgraphs |
| LangGraph Graph API | https://docs.langchain.com/oss/python/langgraph/graph-api |
| LangGraph State & Reducers | https://docs.langchain.com/oss/python/langgraph/use-graph-api |
| LangSmith Background Runs | https://docs.langchain.com/langsmith/agent-server-api/thread-runs/create-background-run |
| LangSmith Webhooks | https://docs.langchain.com/langsmith/use-webhooks |
| Deep Agents Middleware | https://docs.langchain.com/oss/python/deepagents/middleware |
| Deep Agents SubAgent | https://docs.langchain.com/oss/javascript/deepagents/harness |

---

*生成日期: 2025-12-12*
*基于 LangGraph v1 文档*
