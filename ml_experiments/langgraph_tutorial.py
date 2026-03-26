"""
=============================================================================
LangGraph — Zero to Hero Tutorial
=============================================================================

WHAT IS LANGGRAPH?
    LangGraph is a low-level orchestration framework for building stateful,
    multi-step LLM applications. It models agent workflows as directed graphs
    where nodes do work and edges control the flow.

    Think of it as: "A programmable flowchart for AI" — you define the states,
    the processing steps (nodes), and the transitions (edges), including
    conditional branches and cycles.

    LangGraph is built by the LangChain team but can be used independently.

ARCHITECTURE OVERVIEW:

    ┌─────────────────────────────────────────────────────────────┐
    │                     StateGraph                              │
    │                                                             │
    │   START ──► Node A ──► Node B ──┬──► Node C ──► END        │
    │                                 │                           │
    │                                 └──► Node D ──► END        │
    │                                                             │
    │   State: { messages: [...], data: {...} }                   │
    │   (shared TypedDict flowing through every node)             │
    └─────────────────────────────────────────────────────────────┘

KEY CONCEPTS:
    StateGraph     — The graph builder class. Define state, add nodes & edges.
    State          — A TypedDict shared across all nodes. Nodes read from it
                     and return partial updates.
    Node           — A Python function that takes state, does work, returns
                     a dict of state updates.
    Edge           — Connects nodes. Fixed (A→B) or conditional (if X→B, else→C).
    Reducer        — Controls how node outputs merge into state:
                     • Default: overwrite (new value replaces old)
                     • Custom: e.g. operator.add for lists (append, not replace)
    START / END    — Special sentinel nodes for entry and exit points.
    Checkpointer   — Saves state snapshots for memory, replay, and fault tolerance.

CAPABILITIES:
    - Directed graphs with cycles (tool-calling agent loops)
    - Conditional branching (route based on state)
    - Reducers for accumulating state (chat history, logs)
    - Persistence via checkpointers (conversation memory across invocations)
    - Human-in-the-loop (interrupt, inspect, resume)
    - Streaming (token-by-token and event-by-event)
    - Subgraphs (nested graphs for modularity)
    - Dynamic fan-out/fan-in with Send()
    - Works with any LLM via LangChain integrations

LIMITATIONS & MITIGATIONS:
    1. Learning curve: Graph-based thinking is different from linear code.
       → Mitigation: Start with simple linear graphs, add complexity gradually.
    2. Debugging: Hard to trace which node produced which state change.
       → Mitigation: Use the log reducer pattern (Part 3) and LangSmith tracing.
    3. State must be serializable (for checkpointing).
       → Mitigation: Use dicts, lists, strings, Pydantic models. No raw objects.
    4. Conditional edges can create subtle infinite loops.
       → Mitigation: Always set recursion_limit. Add logging to routing functions.
    5. LangChain dependency for LLM integrations (ChatOpenAI, ChatGoogleGenerativeAI).
       → Mitigation: Keep langgraph core logic separate from LLM provider code.
    6. Memory usage grows with conversation length (especially with checkpointing).
       → Mitigation: Use trim_messages or implement state pruning.

SETUP:
    pip install langgraph langchain-google-genai  # or use uv
    export GOOGLE_API_KEY='your-key'

RUN:
    GOOGLE_API_KEY=your-key uv run python ml_experiments/langgraph_tutorial.py

REFERENCE:
    Docs:   https://langchain-ai.github.io/langgraph/
    GitHub: https://github.com/langchain-ai/langgraph
    PyPI:   https://pypi.org/project/langgraph/
=============================================================================
"""

from __future__ import annotations

import os
from operator import add
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ PART 1 — SIMPLEST GRAPH (No LLM needed)                                 ┃
# ┃                                                                          ┃
# ┃ This shows the pure mechanics of LangGraph:                              ┃
# ┃   1. Define a State (TypedDict)                                          ┃
# ┃   2. Write node functions (take state → return partial update)           ┃
# ┃   3. Build the graph (add nodes + edges)                                 ┃
# ┃   4. Compile and invoke                                                  ┃
# ┃                                                                          ┃
# ┃ No API keys needed! Pure Python data processing.                         ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


# --- Step 1: Define State ---
# State is a TypedDict — a typed dictionary that flows through every node.
# Each key is a piece of shared data that any node can read or update.
class TextState(TypedDict):
    text: str          # input text (will be overwritten by nodes)
    word_count: int    # computed by count_words node
    uppercase: str     # computed by to_uppercase node
    reversed_text: str # computed by reverse_text node


# --- Step 2: Define Nodes ---
# A node is a plain function: receives the full state, returns a PARTIAL update.
# Only the keys you return get updated; other keys stay unchanged.
def count_words(state: TextState) -> dict:
    """Count the words in the text.

    Nodes always receive the full state dict and return a partial update.
    Here we only update 'word_count', leaving other keys untouched.
    """
    return {"word_count": len(state["text"].split())}


def to_uppercase(state: TextState) -> dict:
    """Convert the text to uppercase."""
    return {"uppercase": state["text"].upper()}


def reverse_text(state: TextState) -> dict:
    """Reverse the text string."""
    return {"reversed_text": state["text"][::-1]}


# --- Step 3: Build the Graph ---
builder = StateGraph(TextState)

# Add nodes (name → function)
builder.add_node("count_words", count_words)
builder.add_node("to_uppercase", to_uppercase)
builder.add_node("reverse_text", reverse_text)

# Add edges (define execution order)
#   START → count_words → to_uppercase → reverse_text → END
builder.add_edge(START, "count_words")
builder.add_edge("count_words", "to_uppercase")
builder.add_edge("to_uppercase", "reverse_text")
builder.add_edge("reverse_text", END)

# --- Step 4: Compile ---
# compile() freezes the graph into a runnable object.
# After this, no more nodes/edges can be added.
simple_graph = builder.compile()


def demo_simple_graph():
    """Demo: Linear graph processing text without any LLM."""
    print("=" * 70)
    print("  PART 1: Simple Linear Graph (no LLM)")
    print("  Flow: START → count_words → to_uppercase → reverse_text → END")
    print("=" * 70)

    result = simple_graph.invoke({
        "text": "LangGraph makes building AI workflows intuitive",
        "word_count": 0,
        "uppercase": "",
        "reversed_text": "",
    })

    print(f"  Input:     {result['text']}")
    print(f"  Words:     {result['word_count']}")
    print(f"  Upper:     {result['uppercase']}")
    print(f"  Reversed:  {result['reversed_text']}")


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ PART 2 — CONDITIONAL EDGES (Branching)                                   ┃
# ┃                                                                          ┃
# ┃ Sometimes the next step depends on the current state.                    ┃
# ┃ add_conditional_edges() lets you route dynamically.                      ┃
# ┃                                                                          ┃
# ┃ You provide:                                                             ┃
# ┃   1. Source node name                                                    ┃
# ┃   2. A routing function (takes state → returns a string key)             ┃
# ┃   3. A mapping dict {key → target_node_name}                             ┃
# ┃                                                                          ┃
# ┃ FLOW DIAGRAM:                                                           ┃
# ┃                            ┌──► positive ──► END                         ┃
# ┃   START ──► analyze ──────┼──► negative ──► END                         ┃
# ┃                            └──► neutral  ──► END                         ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


class ReviewState(TypedDict):
    text: str          # the review text
    sentiment: str     # detected sentiment
    response: str      # generated response
    confidence: float  # sentiment detection confidence


def analyze_sentiment(state: ReviewState) -> dict:
    """Keyword-based sentiment analysis (no LLM needed for demo).

    In production, you'd call an LLM or a sentiment API here.
    This demonstrates the node pattern with meaningful logic.
    """
    text = state["text"].lower()
    negative_words = {"bad", "terrible", "awful", "hate", "worst", "horrible", "poor", "disappointing"}
    positive_words = {"great", "love", "amazing", "excellent", "best", "wonderful", "fantastic", "perfect"}

    words = set(text.split())
    neg = len(words & negative_words)
    pos = len(words & positive_words)
    total = neg + pos or 1  # avoid division by zero

    if pos > neg:
        return {"sentiment": "positive", "confidence": pos / total}
    elif neg > pos:
        return {"sentiment": "negative", "confidence": neg / total}
    return {"sentiment": "neutral", "confidence": 0.5}


def handle_positive(state: ReviewState) -> dict:
    """Generate response for positive reviews."""
    return {"response": f"Thank you for your wonderful feedback! (confidence: {state['confidence']:.0%})"}


def handle_negative(state: ReviewState) -> dict:
    """Generate response for negative reviews."""
    return {"response": f"We're sorry about your experience. We'll work to improve! (confidence: {state['confidence']:.0%})"}


def handle_neutral(state: ReviewState) -> dict:
    """Generate response for neutral reviews."""
    return {"response": f"Thanks for sharing your thoughts! (confidence: {state['confidence']:.0%})"}


def route_by_sentiment(state: ReviewState) -> str:
    """Routing function: returns a string key matching the conditional edges map.

    This is the core of conditional routing. The returned string must match
    one of the keys in the mapping dict passed to add_conditional_edges().
    """
    return state["sentiment"]


# Build the branching graph
review_builder = StateGraph(ReviewState)
review_builder.add_node("analyze", analyze_sentiment)
review_builder.add_node("positive", handle_positive)
review_builder.add_node("negative", handle_negative)
review_builder.add_node("neutral", handle_neutral)

review_builder.add_edge(START, "analyze")

# Conditional edge: after "analyze", check sentiment to decide next node
review_builder.add_conditional_edges(
    "analyze",            # source node
    route_by_sentiment,   # function that returns a key
    {                     # key → target node mapping
        "positive": "positive",
        "negative": "negative",
        "neutral": "neutral",
    },
)

# All terminal branches lead to END
review_builder.add_edge("positive", END)
review_builder.add_edge("negative", END)
review_builder.add_edge("neutral", END)

review_graph = review_builder.compile()


def demo_conditional_edges():
    """Demo: Branching based on state (conditional routing)."""
    print("\n" + "=" * 70)
    print("  PART 2: Conditional Edges (Dynamic Branching)")
    print("  Flow: START → analyze → [positive|negative|neutral] → END")
    print("=" * 70)

    reviews = [
        "This product is amazing and excellent!",
        "Terrible experience, the worst service ever.",
        "It was okay, nothing special happened.",
    ]

    for review in reviews:
        result = review_graph.invoke({
            "text": review, "sentiment": "", "response": "", "confidence": 0.0,
        })
        print(f"\n  Review:     {result['text']}")
        print(f"  Sentiment:  {result['sentiment']} ({result['confidence']:.0%})")
        print(f"  Response:   {result['response']}")


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ PART 3 — REDUCERS (Accumulating State)                                   ┃
# ┃                                                                          ┃
# ┃ BY DEFAULT: returning {"key": value} OVERWRITES the existing value.      ┃
# ┃ WITH REDUCER: values ACCUMULATE instead of overwriting.                  ┃
# ┃                                                                          ┃
# ┃ Syntax: Annotated[type, reducer_function]                                ┃
# ┃                                                                          ┃
# ┃ Common reducers:                                                         ┃
# ┃   operator.add  — for lists: appends items; for ints: sums them          ┃
# ┃   add_messages   — LangGraph built-in for chat messages (deduplicates)   ┃
# ┃   custom func   — any (old, new) → merged function                      ┃
# ┃                                                                          ┃
# ┃ WHY THIS MATTERS:                                                        ┃
# ┃   Without reducers, a chat history would be lost after each node.        ┃
# ┃   With add_messages reducer, each node APPENDS to the conversation       ┃
# ┃   instead of replacing it. This is foundational for agent loops.         ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


class PipelineState(TypedDict):
    data: str                              # overwrites (default reducer)
    log: Annotated[list[str], add]         # APPENDS (operator.add reducer)
    error_count: Annotated[int, add]       # SUMS (operator.add on ints)


def step_validate(state: PipelineState) -> dict:
    """Validate input data."""
    is_valid = len(state["data"]) > 0
    return {
        "log": [f"validate: input={'valid' if is_valid else 'empty'}"],
        "error_count": 0 if is_valid else 1,
    }


def step_transform(state: PipelineState) -> dict:
    """Transform the data (uppercase + trim)."""
    return {
        "data": state["data"].strip().upper(),
        "log": ["transform: uppercased and trimmed"],
    }


def step_enrich(state: PipelineState) -> dict:
    """Enrich the data with metadata."""
    return {
        "data": f"[PROCESSED] {state['data']}",
        "log": ["enrich: added [PROCESSED] prefix"],
    }


pipeline_builder = StateGraph(PipelineState)
pipeline_builder.add_node("validate", step_validate)
pipeline_builder.add_node("transform", step_transform)
pipeline_builder.add_node("enrich", step_enrich)
pipeline_builder.add_edge(START, "validate")
pipeline_builder.add_edge("validate", "transform")
pipeline_builder.add_edge("transform", "enrich")
pipeline_builder.add_edge("enrich", END)
pipeline_graph = pipeline_builder.compile()


def demo_reducers():
    """Demo: Reducers for accumulating state across nodes."""
    print("\n" + "=" * 70)
    print("  PART 3: Reducers (Accumulating State)")
    print("  Flow: START → validate → transform → enrich → END")
    print("  Key insight: 'log' list ACCUMULATES across all nodes!")
    print("=" * 70)

    result = pipeline_graph.invoke({
        "data": "  hello world  ",
        "log": [],
        "error_count": 0,
    })

    print(f"  Data:        {result['data']}")
    print(f"  Errors:      {result['error_count']}")
    print(f"  Log (accumulated):")
    for entry in result["log"]:
        print(f"    - {entry}")


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ PART 4 — CYCLES (Iterative Agent Loops)                                  ┃
# ┃                                                                          ┃
# ┃ LangGraph's SUPERPOWER: cycles. Unlike DAG frameworks, LangGraph was     ┃
# ┃ designed specifically for loops. This is how tool-calling agents work:    ┃
# ┃                                                                          ┃
# ┃   START → agent → tool → agent → tool → ... → END                       ┃
# ┃                    ▲                │                                     ┃
# ┃                    └────────────────┘  (cycle)                           ┃
# ┃                                                                          ┃
# ┃ The cycle continues until a conditional edge routes to END.              ┃
# ┃                                                                          ┃
# ┃ SAFETY: Set recursion_limit to prevent infinite loops (default: 1000).   ┃
# ┃ The limit counts "super-steps" (each node execution = 1 step).          ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


class LoopState(TypedDict):
    counter: int
    max_count: int
    log: Annotated[list[str], add]


def increment(state: LoopState) -> dict:
    """Increment the counter by 1."""
    new_val = state["counter"] + 1
    return {
        "counter": new_val,
        "log": [f"counter incremented to {new_val}"],
    }


def should_continue(state: LoopState) -> str:
    """Routing function: loop back or exit.

    Returns "loop" to cycle back to increment, or "done" to exit.
    This is the exit condition for the cycle.
    """
    if state["counter"] >= state["max_count"]:
        return "done"
    return "loop"


loop_builder = StateGraph(LoopState)
loop_builder.add_node("increment", increment)
loop_builder.add_edge(START, "increment")
loop_builder.add_conditional_edges(
    "increment",
    should_continue,
    {
        "loop": "increment",  # ← CYCLE: points back to itself!
        "done": END,
    },
)
loop_graph = loop_builder.compile()


def demo_cycles():
    """Demo: Cyclic graph that loops until a condition is met."""
    print("\n" + "=" * 70)
    print("  PART 4: Cycles (Iterative Loops)")
    print("  Flow: START → increment ──┐")
    print("                  ▲         │ (loop until counter >= max)")
    print("                  └─────────┘")
    print("=" * 70)

    result = loop_graph.invoke(
        {"counter": 0, "max_count": 5, "log": []},
        config={"recursion_limit": 25},  # safety limit
    )

    print(f"  Final counter: {result['counter']}")
    print(f"  Iterations:")
    for entry in result["log"]:
        print(f"    - {entry}")


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ PART 5 — LLM TOOL-CALLING AGENT (ReAct Pattern)                         ┃
# ┃                                                                          ┃
# ┃ The most common LangGraph pattern. An LLM decides whether to:            ┃
# ┃   a) Call a tool (→ execute tool → loop back to LLM), or                ┃
# ┃   b) Respond directly (→ END)                                           ┃
# ┃                                                                          ┃
# ┃ FLOW:                                                                    ┃
# ┃   START → agent → [has tool_calls?] ──yes──► tools → agent (loop)       ┃
# ┃                          │                                               ┃
# ┃                          no ──► END                                      ┃
# ┃                                                                          ┃
# ┃ COMPONENTS:                                                              ┃
# ┃   - ChatModel.bind_tools(tools) → LLM knows which tools are available   ┃
# ┃   - ToolNode (prebuilt) → executes tool calls from LLM response         ┃
# ┃   - add_messages reducer → accumulates the full conversation             ┃
# ┃   - Conditional edge → checks for tool_calls in LLM response            ┃
# ┃                                                                          ┃
# ┃ REQUIRES: GOOGLE_API_KEY (or OPENAI_API_KEY if using OpenAI)             ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

from langchain_core.tools import tool  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402


class AgentState(TypedDict):
    """State for the tool-calling agent.

    The messages list uses add_messages reducer which:
    - Appends new messages (doesn't overwrite)
    - Deduplicates by message ID
    - Handles the full conversation history automatically
    """

    messages: Annotated[list, add_messages]


# --- Define tools with @tool decorator ---
# LangChain's @tool decorator creates a structured tool with name, description,
# and argument schema derived from the function signature and docstring.

@tool
def search_knowledge(query: str) -> str:
    """Search an internal knowledge base for information.

    Use this tool when the user asks factual questions about
    programming frameworks, languages, or tools.

    Args:
        query: The search query.
    """
    kb = {
        "langgraph": "LangGraph is a framework for building stateful AI agents using directed graphs. "
        "It supports cycles, conditional edges, persistence, and human-in-the-loop patterns.",
        "langchain": "LangChain is a framework for building LLM-powered applications. "
        "It provides abstractions for chains, prompts, memory, and tool integrations.",
        "google adk": "Google ADK (Agent Development Kit) is an open-source framework for "
        "building AI agents, optimized for Gemini but model-agnostic.",
        "python": "Python is a high-level programming language known for readability "
        "and a vast ecosystem of libraries for data science, web, and AI.",
        "react pattern": "ReAct (Reasoning + Acting) is an agent pattern where the LLM "
        "alternates between thinking (reasoning) and acting (calling tools).",
    }
    for key, value in kb.items():
        if key in query.lower():
            return value
    return f"No results found for: '{query}'. Try: langgraph, langchain, google adk, python, react pattern."


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Only supports basic arithmetic: +, -, *, /, parentheses.

    Args:
        expression: A math expression like '2 + 3 * 4' or '(10 + 5) / 3'.
    """
    allowed = set("0123456789+-*/.(). ")
    if not all(c in allowed for c in expression):
        return "Error: only basic arithmetic is supported (+, -, *, /, parentheses)"
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


@tool
def get_current_date() -> str:
    """Get today's date. No arguments needed."""
    import datetime

    return datetime.date.today().isoformat()


agent_tools = [search_knowledge, calculate, get_current_date]


def build_react_agent():
    """Build a ReAct-style tool-calling agent from scratch.

    This function demonstrates the full manual construction of a tool-calling
    agent. For production use, consider create_react_agent() (Part 7).

    Returns:
        Compiled LangGraph that implements the ReAct pattern.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    # Create LLM and bind tools to it
    # bind_tools() tells the LLM about available tools and their schemas
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    llm_with_tools = llm.bind_tools(agent_tools)

    # Node: invoke the LLM
    def call_llm(state: AgentState) -> dict:
        """Call the LLM with the full conversation history.

        The LLM will either:
        a) Return a response with tool_calls → we need to execute them
        b) Return a plain text response → we're done
        """
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    # Node: execute tool calls
    # ToolNode is a prebuilt helper that:
    # 1. Reads tool_calls from the last message
    # 2. Executes each tool
    # 3. Returns ToolMessage(s) with results
    tool_node = ToolNode(agent_tools)

    # Routing function: check if the LLM wants to call tools
    def should_call_tool(state: AgentState) -> str:
        """Check the last message for tool calls.

        If the LLM response contains tool_calls, route to "tools" node.
        Otherwise, route to END (the agent is done).
        """
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "end"

    # Build the graph
    builder = StateGraph(AgentState)
    builder.add_node("agent", call_llm)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_call_tool, {"tools": "tools", "end": END})
    builder.add_edge("tools", "agent")  # THE CYCLE: tool results go back to LLM

    return builder.compile()


def demo_tool_agent():
    """Demo: Full ReAct agent with tool calling."""
    print("\n" + "=" * 70)
    print("  PART 5: LLM Tool-Calling Agent (ReAct Pattern)")
    print("  Flow: START → agent ⟷ tools → END")
    print("=" * 70)

    if not os.environ.get("GOOGLE_API_KEY"):
        print("  SKIPPED — set GOOGLE_API_KEY to run this demo.")
        return

    graph = build_react_agent()

    questions = [
        "What is LangGraph? Give me a brief summary.",
        "What is 42 * 17 + 3?",
        "What is today's date?",
    ]

    for q in questions:
        print(f"\n  USER: {q}")
        result = graph.invoke({"messages": [("user", q)]})
        last_msg = result["messages"][-1]
        print(f"  AGENT: {last_msg.content}")


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ PART 6 — PERSISTENCE (Conversation Memory)                               ┃
# ┃                                                                          ┃
# ┃ By default, each invoke() is stateless — no memory between calls.        ┃
# ┃ Compile with a checkpointer to get:                                      ┃
# ┃   - Conversation memory across multiple invocations                      ┃
# ┃   - Time-travel debugging (replay from any checkpoint)                   ┃
# ┃   - Fault tolerance (resume from last successful step)                   ┃
# ┃   - Human-in-the-loop (pause → inspect → approve → resume)              ┃
# ┃                                                                          ┃
# ┃ Each "thread_id" maintains its own independent state history.            ┃
# ┃                                                                          ┃
# ┃ Checkpointer options:                                                    ┃
# ┃   InMemorySaver          — dev/testing (data lost on restart)            ┃
# ┃   SqliteSaver            — local persistence                             ┃
# ┃   PostgresSaver          — production persistence                        ┃
# ┃   Custom BaseCheckpointer — bring your own storage                       ┃
# ┃                                                                          ┃
# ┃ LIMITATION: InMemorySaver loses data on restart.                         ┃
# ┃ MITIGATION: Use SqliteSaver/PostgresSaver for anything beyond testing.   ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402


def build_persistent_agent():
    """Build a ReAct agent with conversation memory."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0).bind_tools(agent_tools)

    def call_llm(state: AgentState) -> dict:
        return {"messages": [llm.invoke(state["messages"])]}

    tool_node = ToolNode(agent_tools)

    def should_call_tool(state: AgentState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "end"

    builder = StateGraph(AgentState)
    builder.add_node("agent", call_llm)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_call_tool, {"tools": "tools", "end": END})
    builder.add_edge("tools", "agent")

    # KEY: compile with a checkpointer!
    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)


def demo_persistence():
    """Demo: Conversation memory across multiple invocations."""
    print("\n" + "=" * 70)
    print("  PART 6: Persistence (Conversation Memory)")
    print("  Key: same thread_id = same conversation")
    print("=" * 70)

    if not os.environ.get("GOOGLE_API_KEY"):
        print("  SKIPPED — set GOOGLE_API_KEY to run this demo.")
        return

    graph = build_persistent_agent()

    # Thread 1: multi-turn conversation
    config = {"configurable": {"thread_id": "thread-1"}}

    print("\n  --- Thread 1, Turn 1 ---")
    r1 = graph.invoke({"messages": [("user", "What is LangGraph?")]}, config)
    print(f"  AGENT: {r1['messages'][-1].content[:200]}...")

    print("\n  --- Thread 1, Turn 2 (agent remembers!) ---")
    r2 = graph.invoke({"messages": [("user", "How does it handle state?")]}, config)
    print(f"  AGENT: {r2['messages'][-1].content[:200]}...")

    # Thread 2: fresh conversation (no memory of thread-1)
    config2 = {"configurable": {"thread_id": "thread-2"}}
    print("\n  --- Thread 2 (new thread = no memory) ---")
    r3 = graph.invoke({"messages": [("user", "What were we just discussing?")]}, config2)
    print(f"  AGENT: {r3['messages'][-1].content[:200]}...")

    # Inspect state (powerful debugging tool)
    print("\n  --- Inspecting Thread 1 State ---")
    snapshot = graph.get_state(config)
    print(f"  Total messages in thread-1: {len(snapshot.values['messages'])}")
    print(f"  Next node: {snapshot.next}")  # empty tuple = graph is done


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ PART 7 — PREBUILT AGENT (One-Liner)                                      ┃
# ┃                                                                          ┃
# ┃ For quick prototyping, langgraph.prebuilt provides create_react_agent()  ┃
# ┃ which builds the entire ReAct graph in one call.                         ┃
# ┃                                                                          ┃
# ┃ Equivalent to Part 5 but without manual graph construction.              ┃
# ┃ Good for prototyping, but build manually when you need customization.    ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

from langgraph.prebuilt import create_react_agent  # noqa: E402


def demo_prebuilt_agent():
    """Demo: One-liner prebuilt ReAct agent."""
    print("\n" + "=" * 70)
    print("  PART 7: Prebuilt Agent (create_react_agent)")
    print("  One line to create a fully functional tool-calling agent!")
    print("=" * 70)

    if not os.environ.get("GOOGLE_API_KEY"):
        print("  SKIPPED — set GOOGLE_API_KEY to run this demo.")
        return

    # One line! Builds the full ReAct graph under the hood.
    agent = create_react_agent(
        "google_genai:gemini-2.5-flash",
        tools=agent_tools,
        prompt="You are a helpful assistant. Use your tools when appropriate. Be concise.",
    )

    result = agent.invoke({"messages": [("user", "What's 123 * 456?")]})
    print(f"\n  USER:  What's 123 * 456?")
    print(f"  AGENT: {result['messages'][-1].content}")


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ PART 8 — STREAMING (Real-Time Output)                                    ┃
# ┃                                                                          ┃
# ┃ LangGraph supports two streaming modes:                                  ┃
# ┃   stream()  — yields state updates after each node ("super-step")        ┃
# ┃   astream_events() — yields individual events (tokens, tool calls, etc.) ┃
# ┃                                                                          ┃
# ┃ stream() modes:                                                          ┃
# ┃   "values"  — full state after each node                                 ┃
# ┃   "updates" — only the delta (what changed)                              ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


def demo_streaming():
    """Demo: Streaming state updates from a graph."""
    print("\n" + "=" * 70)
    print("  PART 8: Streaming (Real-Time State Updates)")
    print("  Uses the simple pipeline graph from Part 3")
    print("=" * 70)

    print("\n  Streaming 'updates' mode (shows only what changed per step):")
    for update in pipeline_graph.stream(
        {"data": "  streaming demo  ", "log": [], "error_count": 0},
        stream_mode="updates",
    ):
        # update is a dict: {node_name: {state_updates}}
        for node_name, state_update in update.items():
            print(f"    [{node_name}] → {state_update}")


# ===========================================================================
# MAIN — Run all demos
# ===========================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  LANGGRAPH — Zero to Hero Tutorial")
    print("  Running all demos...")
    print("=" * 70)

    # Parts 1-4: No API key needed (pure Python logic)
    demo_simple_graph()
    demo_conditional_edges()
    demo_reducers()
    demo_cycles()
    demo_streaming()

    # Parts 5-7: Need GOOGLE_API_KEY
    demo_tool_agent()
    demo_persistence()
    demo_prebuilt_agent()

    print("\n\n" + "=" * 70)
    print("  ALL DEMOS COMPLETE!")
    print("=" * 70)
    print("""
    WHAT YOU LEARNED:
      1. StateGraph  — Define state schema, add nodes & edges, compile
      2. Nodes       — Python functions: take state, return partial updates
      3. Edges       — Fixed (A→B) and conditional (if X→B else→C)
      4. Reducers    — operator.add to accumulate lists; add_messages for chat
      5. Cycles      — Nodes can loop back (tool-calling agent pattern)
      6. ReAct Agent — LLM → tools → LLM → ... → END (the core agent loop)
      7. Persistence — Checkpointers for conversation memory across calls
      8. Prebuilt    — create_react_agent() for quick prototyping
      9. Streaming   — Real-time state updates with stream()

    CHEAT SHEET:
      from langgraph.graph import StateGraph, START, END
      builder = StateGraph(MyState)
      builder.add_node("name", func)
      builder.add_edge("a", "b")
      builder.add_conditional_edges("src", route_fn, {"key": "node"})
      graph = builder.compile(checkpointer=InMemorySaver())
      result = graph.invoke(state, {"configurable": {"thread_id": "t1"}})

    NEXT STEPS:
      - Human-in-the-loop: compile(interrupt_before=["node"])
      - Subgraphs: nest StateGraphs for modularity
      - Dynamic fan-out: Send() for map-reduce patterns
      - LangSmith: add tracing for debugging complex graphs
      - Read: https://langchain-ai.github.io/langgraph/
    """)
