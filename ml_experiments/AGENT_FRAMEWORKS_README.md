# Agent Frameworks — Zero to Hero Tutorials

Three comprehensive, self-contained Python tutorials covering the major AI agent/LLM frameworks.

## Quick Start

```bash
# Install dependencies
uv sync --extra agents

# Set your API key (get one at https://aistudio.google.com/apikey)
export GOOGLE_API_KEY='your-key-here'

# Run any tutorial
uv run python ml_experiments/langchain_tutorial.py    # Start here (foundation)
uv run python ml_experiments/langgraph_tutorial.py    # Then this (agents)
uv run python ml_experiments/google_adk_tutorial.py   # Then this (alternative)
```

## Recommended Learning Order

```
1. LangChain (foundation)  →  2. LangGraph (agents)  →  3. Google ADK (alternative)
```

## Tutorial Overview

### 1. `langchain_tutorial.py` — LangChain (The Foundation)

**What it is:** A standard library for LLM applications — models, prompts, parsers, tools, chains.

| Part | Topic | LLM Required? | What You Learn |
|------|-------|---------------|----------------|
| 1 | Chat Models | Yes | Invoke, stream, batch with any LLM provider |
| 2 | Prompt Templates | Yes | Parameterized prompts, LCEL composition |
| 3 | Output Parsers | Yes | Structured output with Pydantic, StrOutputParser |
| 4 | LCEL | Yes | Pipe syntax, RunnableParallel, RunnableLambda, batch |
| 5 | Tools | Yes | @tool decorator, bind_tools, manual tool execution |
| 6 | RAG | Yes | Retrieval-Augmented Generation (manual approach) |
| 7 | Advanced LCEL | Yes | Multi-step chains, passthrough, data transformation |

**Key functions & classes:**

| Name | What it does | Limitations |
|------|-------------|-------------|
| `ChatGoogleGenerativeAI` | Wraps Gemini API | Requires API key; rate limits apply |
| `ChatPromptTemplate` | Parameterized prompt builder | Variables must match at invoke time |
| `StrOutputParser` | Extracts `.content` from AIMessage | Only returns text, no metadata |
| `.with_structured_output()` | Forces JSON output matching a Pydantic schema | Requires tool-calling-capable model |
| `@tool` decorator | Creates a tool from a function | Docstring quality affects LLM's ability to use it |
| `RunnableParallel` | Runs multiple chains simultaneously | All branches must accept same input shape |
| `.batch()` | Processes multiple inputs in parallel | May hit rate limits with large batches |

---

### 2. `langgraph_tutorial.py` — LangGraph (Agent Orchestration)

**What it is:** A graph-based framework for building stateful AI agents with cycles, branches, and persistence.

| Part | Topic | LLM Required? | What You Learn |
|------|-------|---------------|----------------|
| 1 | Simple Graph | No | StateGraph, nodes, edges, compile, invoke |
| 2 | Conditional Edges | No | Dynamic routing based on state |
| 3 | Reducers | No | Accumulating state (append vs overwrite) |
| 4 | Cycles | No | Iterative loops with exit conditions |
| 5 | ReAct Agent | Yes | LLM tool-calling agent (full manual build) |
| 6 | Persistence | Yes | Conversation memory with checkpointers |
| 7 | Prebuilt Agent | Yes | One-liner `create_react_agent()` |
| 8 | Streaming | No | Real-time state updates with `stream()` |

**Key functions & classes:**

| Name | What it does | Limitations |
|------|-------------|-------------|
| `StateGraph` | Graph builder — add nodes, edges, compile | State must be serializable (TypedDict/Pydantic) |
| `START` / `END` | Entry and exit sentinel nodes | Every graph needs at least START → ... → END |
| `.add_node(name, fn)` | Register a processing step | Node names must be unique |
| `.add_edge(a, b)` | Fixed routing between nodes | Cannot be conditional |
| `.add_conditional_edges()` | Dynamic routing based on state | Routing function must return a valid key |
| `Annotated[list, add]` | Reducer: append instead of overwrite | Reducer must match the type (add for lists/ints) |
| `add_messages` | Built-in chat message reducer | Specific to message lists |
| `ToolNode` | Prebuilt node that executes tool calls | Expects messages with `.tool_calls` |
| `InMemorySaver` | In-memory checkpointer for persistence | Data lost on restart |
| `create_react_agent()` | One-liner prebuilt ReAct agent | Less customizable than manual build |
| `.compile()` | Freezes graph into runnable | No modifications after compile |
| `recursion_limit` | Safety limit for cycles | Default 1000; set lower for debugging |

---

### 3. `google_adk_tutorial.py` — Google ADK (Agent Development Kit)

**What it is:** Google's open-source framework for building AI agents, optimized for Gemini.

| Part | Topic | LLM Required? | What You Learn |
|------|-------|---------------|----------------|
| 1 | Tools | No (definition only) | Python functions as tools, best practices |
| 2 | Agents | Yes (at runtime) | Agent constructor, instruction, description |
| 3 | Structured Output | Yes | output_schema with Pydantic models |
| 4 | Runner & Sessions | Yes | InMemoryRunner, event streaming, sessions |
| 5 | Multi-Agent | Yes | Coordinator routing to specialists |
| 6 | Workflow Agents | Yes | Sequential, Parallel, Loop agents |
| 7 | Session State | Yes | output_key for cross-agent data sharing |
| 8 | Callbacks | Yes | before_tool_callback for logging/validation |

**Key functions & classes:**

| Name | What it does | Limitations |
|------|-------------|-------------|
| `Agent` / `LlmAgent` | LLM-powered agent with tools & sub-agents | Requires Gemini API key |
| `SequentialAgent` | Runs sub-agents in order | No conditional logic (deterministic) |
| `ParallelAgent` | Runs sub-agents concurrently | Results merged into same conversation |
| `LoopAgent` | Iterates sub-agents until done | Needs `max_iterations` to prevent infinite loops |
| `InMemoryRunner` | Executes agents, streams events | Data lost on restart |
| `output_schema` | Forces structured JSON output | Complex schemas may confuse the LLM |
| `output_key` | Saves agent response to session state | Only saves final response text |
| `sub_agents` | Child agents for multi-agent routing | Routing quality depends on description clarity |
| `before_tool_callback` | Intercepts tool calls | Return None to proceed, dict to short-circuit |
| `genai_types.Content` | Message format (shared with Gemini SDK) | Must use Parts for content |

---

## Comparison: When to Use What

| Scenario | Recommended | Why |
|----------|-------------|-----|
| Simple LLM call (prompt → response) | LangChain | Minimal setup, clean abstractions |
| Structured output / parsing | LangChain | `with_structured_output()` is simple |
| RAG pipeline | LangChain | Built-in retriever + chain patterns |
| Stateful agent with tool loops | LangGraph | Cycles + persistence + conditional routing |
| Multi-agent coordination | Google ADK or LangGraph | ADK if Gemini-first; LangGraph if provider-agnostic |
| Deterministic workflow (A→B→C) | Google ADK | SequentialAgent is simpler than LangGraph for fixed flows |
| Production agent with memory | LangGraph | Checkpointers + thread-based memory |
| Quick prototype | Google ADK (`adk web`) | Built-in dev UI + CLI |
| Maximum control over flow | LangGraph | Graph-level control over every transition |

## Common Issues & Solutions

| Problem | Solution |
|---------|----------|
| `GOOGLE_API_KEY` not set | `export GOOGLE_API_KEY='your-key'` |
| Rate limiting (429 errors) | Add retry logic or reduce batch size |
| Tool not being called | Improve the tool's docstring — LLMs read it |
| Wrong sub-agent selected (ADK) | Make agent descriptions more distinct |
| Infinite loop (LangGraph) | Lower `recursion_limit`, add logging to routing function |
| State not persisting | Ensure same `thread_id` + compile with checkpointer |
| Import errors | Run `uv sync --extra agents` to install dependencies |
| Streaming not working | Use `.stream()` instead of `.invoke()` |

## Dependencies

All managed via the `agents` optional dependency group in `pyproject.toml`:

```toml
[project.optional-dependencies]
agents = [
    "google-adk>=1.27.0",
    "langgraph>=1.1.0",
    "langchain-google-genai>=2.1.0",
    "langchain>=1.2.12",
    "langchain-community>=0.4.1",
]
```

Install with: `uv sync --extra agents`
