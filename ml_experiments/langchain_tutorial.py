"""
=============================================================================
LangChain — Zero to Hero Tutorial
=============================================================================

WHAT IS LANGCHAIN?
    LangChain is a framework for building applications powered by language
    models. It provides modular abstractions for the common components of
    LLM applications: prompts, models, output parsers, chains, memory,
    tools, retrievers, and agents.

    Think of it as: "A standard library for LLM applications" — it gives
    you building blocks that compose together cleanly.

    LangChain is the FOUNDATION layer. LangGraph (for agents/workflows)
    and LangSmith (for observability) build on top of it.

ECOSYSTEM:
    langchain-core    — Base abstractions (Runnables, prompts, messages)
    langchain         — Chains, agents, retrieval strategies
    langchain-community — 3rd party integrations (maintained by community)
    langchain-google-genai — Google Gemini integration
    langchain-openai  — OpenAI integration
    langgraph         — Stateful agent orchestration (separate tutorial)
    langsmith         — Tracing, evaluation, monitoring

ARCHITECTURE:

    ┌──────────────────────────────────────────────────────────┐
    │                    LangChain App                         │
    │                                                          │
    │  Prompt ──► Model ──► Output Parser ──► Result           │
    │    │          │            │                              │
    │    │     ┌────┴────┐      │                              │
    │    │     │  Tools   │      │                              │
    │    │     │  Memory  │      │                              │
    │    │     │ Retriever│      │                              │
    │    │     └──────────┘      │                              │
    │    │                       │                              │
    │    └──── Chain (LCEL) ─────┘                              │
    └──────────────────────────────────────────────────────────┘

KEY CONCEPTS:
    Runnable       — The base interface. Everything in LangChain is a Runnable
                     with .invoke(), .stream(), .batch() methods.
    LCEL           — LangChain Expression Language. Pipe syntax for composing
                     Runnables: prompt | model | parser (like Unix pipes).
    ChatModel      — Wrapper around an LLM (Gemini, GPT, Claude, etc.)
    PromptTemplate — Template for constructing prompts with variables.
    OutputParser   — Parses LLM output into structured data.
    Tool           — A function callable by an LLM with schema metadata.
    Chain          — A sequence of Runnables composed with LCEL (|).
    Retriever      — Fetches relevant documents for context (RAG).
    Memory         — Stores conversation history (deprecated in favor of
                     LangGraph's persistence, but still useful for simple cases).

CAPABILITIES:
    - Chat with any LLM provider (Gemini, OpenAI, Anthropic, local models)
    - Structured output (Pydantic models, JSON schemas)
    - Tool/function calling with @tool decorator
    - RAG (Retrieval-Augmented Generation) pipelines
    - LCEL for composable, streamable chains
    - Batch processing with automatic parallelism
    - Streaming (token-by-token and event-by-event)
    - Fallbacks and retries
    - Caching (in-memory, Redis, SQLite)

LIMITATIONS & MITIGATIONS:
    1. Abstraction overhead: Many layers between you and the LLM API.
       → Mitigation: Use langchain-core directly for simple cases. Drop down
         to the raw SDK (google.genai) when LangChain adds no value.
    2. Rapid API changes: Breaking changes between versions are common.
       → Mitigation: Pin versions. Read migration guides before upgrading.
    3. Provider quirks: Not all providers support all features uniformly.
       → Mitigation: Test with your specific provider. Check provider docs.
    4. Memory (ConversationBufferMemory etc.) is being deprecated.
       → Mitigation: Use LangGraph checkpointers for stateful conversations.
    5. Error messages can be cryptic (deep call stacks).
       → Mitigation: Use LangSmith tracing to see what's happening internally.
    6. Import paths change frequently.
       → Mitigation: Follow official docs for current import paths.

SETUP:
    pip install langchain langchain-google-genai  # or use uv
    export GOOGLE_API_KEY='your-key'

RUN:
    GOOGLE_API_KEY=your-key uv run python ml_experiments/langchain_tutorial.py

REFERENCE:
    Docs:   https://python.langchain.com/docs/
    GitHub: https://github.com/langchain-ai/langchain
    PyPI:   https://pypi.org/project/langchain/
=============================================================================
"""

from __future__ import annotations

import os

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ PART 1 — CHAT MODELS (The Foundation)                                    ┃
# ┃                                                                          ┃
# ┃ A ChatModel wraps an LLM provider's API. It takes messages in, returns   ┃
# ┃ messages out. All ChatModels share the same interface:                    ┃
# ┃   .invoke(messages) → AIMessage                                          ┃
# ┃   .stream(messages) → Iterator[AIMessageChunk]                           ┃
# ┃   .batch([msgs1, msgs2]) → [AIMessage, AIMessage]                       ┃
# ┃                                                                          ┃
# ┃ This uniformity means you can swap providers without changing app code.  ┃
# ┃                                                                          ┃
# ┃ Message types:                                                           ┃
# ┃   SystemMessage  — instructions for the model (persona, rules)           ┃
# ┃   HumanMessage   — user's input                                          ┃
# ┃   AIMessage      — model's response                                      ┃
# ┃   ToolMessage    — result from a tool call                               ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402


def demo_chat_model():
    """Demo: Basic chat model usage with different message types."""
    print("=" * 70)
    print("  PART 1: Chat Models (The Foundation)")
    print("=" * 70)

    if not os.environ.get("GOOGLE_API_KEY"):
        print("  SKIPPED — set GOOGLE_API_KEY to run this demo.")
        return

    from langchain_google_genai import ChatGoogleGenerativeAI

    # Create a chat model
    # Key parameters:
    #   model       — model ID (e.g. "gemini-2.5-flash")
    #   temperature — 0 = deterministic, 1 = creative (default varies)
    #   max_tokens  — max output length
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    # --- Method 1: Simple string input ---
    print("\n  Method 1: Simple string")
    response = llm.invoke("What is Python in one sentence?")
    print(f"  Response: {response.content}")

    # --- Method 2: Structured messages ---
    print("\n  Method 2: Structured messages")
    messages = [
        SystemMessage(content="You are a concise technical writer. Answer in exactly one sentence."),
        HumanMessage(content="What is LangChain?"),
    ]
    response = llm.invoke(messages)
    print(f"  Response: {response.content}")

    # --- Method 3: Tuple shorthand ---
    print("\n  Method 3: Tuple shorthand (most common)")
    response = llm.invoke([
        ("system", "You answer in haiku format (5-7-5 syllables)."),
        ("human", "What is machine learning?"),
    ])
    print(f"  Response:\n    {response.content}")

    # --- Method 4: Streaming ---
    print("\n  Method 4: Streaming (token by token)")
    print("  Response: ", end="")
    for chunk in llm.stream("Count from 1 to 5, one per line."):
        print(chunk.content, end="", flush=True)
    print()


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ PART 2 — PROMPT TEMPLATES                                                ┃
# ┃                                                                          ┃
# ┃ PromptTemplates let you parameterize prompts with variables.             ┃
# ┃ They're Runnables, so they compose with the | pipe operator.             ┃
# ┃                                                                          ┃
# ┃ Types:                                                                   ┃
# ┃   ChatPromptTemplate  — for chat models (produces messages)              ┃
# ┃   PromptTemplate      — for completion models (produces a string)        ┃
# ┃   FewShotPromptTemplate — includes examples for few-shot learning        ┃
# ┃                                                                          ┃
# ┃ BEST PRACTICE: Always use templates instead of f-strings. Templates      ┃
# ┃ are serializable, versionable, and compose with LCEL.                    ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

from langchain_core.prompts import ChatPromptTemplate  # noqa: E402


def demo_prompt_templates():
    """Demo: Parameterized prompt templates."""
    print("\n" + "=" * 70)
    print("  PART 2: Prompt Templates")
    print("=" * 70)

    if not os.environ.get("GOOGLE_API_KEY"):
        print("  SKIPPED — set GOOGLE_API_KEY to run this demo.")
        return

    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    # --- Basic template ---
    print("\n  Basic ChatPromptTemplate:")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in {domain}. Answer concisely in 1-2 sentences."),
        ("human", "{question}"),
    ])

    # Inspect what variables the template expects
    print(f"  Template variables: {prompt.input_variables}")

    # Format the prompt (produces a list of messages)
    formatted = prompt.invoke({"domain": "Python", "question": "What are decorators?"})
    print(f"  Formatted: {formatted.messages}")

    # --- LCEL Chain: prompt | model ---
    # The | operator composes Runnables: output of left feeds into right
    chain = prompt | llm
    response = chain.invoke({"domain": "databases", "question": "What is an index?"})
    print(f"\n  Chain response: {response.content}")

    # --- Template with multiple user turns ---
    print("\n  Multi-turn template:")
    multi_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful tutor."),
        ("human", "I'm learning about {topic}."),
        ("ai", "Great! What would you like to know about {topic}?"),
        ("human", "{followup}"),
    ])
    chain = multi_prompt | llm
    response = chain.invoke({"topic": "recursion", "followup": "Give me a one-line analogy."})
    print(f"  Response: {response.content}")


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ PART 3 — OUTPUT PARSERS (Structured Output)                              ┃
# ┃                                                                          ┃
# ┃ LLMs return strings. Output parsers transform them into structured data. ┃
# ┃                                                                          ┃
# ┃ Two approaches:                                                          ┃
# ┃   1. with_structured_output() — uses the model's native tool calling     ┃
# ┃      to guarantee valid structured output. PREFERRED.                    ┃
# ┃   2. Output parsers — prompt-based parsing (StrOutputParser, JSON, etc.) ┃
# ┃      Less reliable but works with any model.                             ┃
# ┃                                                                          ┃
# ┃ LIMITATION: with_structured_output() requires a model that supports      ┃
# ┃ tool calling. Most modern models do (Gemini, GPT-4, Claude).             ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402


class MovieReview(BaseModel):
    """Structured output schema for movie reviews."""

    title: str = Field(description="The movie title")
    rating: float = Field(description="Rating from 1.0 to 10.0")
    genre: str = Field(description="Primary genre")
    one_line_summary: str = Field(description="One sentence summary")
    would_recommend: bool = Field(description="Whether you'd recommend this movie")


def demo_output_parsers():
    """Demo: Structured output with Pydantic models and LCEL."""
    print("\n" + "=" * 70)
    print("  PART 3: Output Parsers (Structured Output)")
    print("=" * 70)

    if not os.environ.get("GOOGLE_API_KEY"):
        print("  SKIPPED — set GOOGLE_API_KEY to run this demo.")
        return

    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    # --- Method 1: StrOutputParser (extract just the text) ---
    print("\n  Method 1: StrOutputParser (text only)")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a poet. Write exactly one line of poetry."),
        ("human", "Write about {topic}."),
    ])
    # prompt | llm returns AIMessage; | StrOutputParser() extracts .content
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"topic": "code"})
    print(f"  Result (str): {result}")
    print(f"  Type: {type(result).__name__}")  # str, not AIMessage!

    # --- Method 2: with_structured_output (Pydantic model) ---
    print("\n  Method 2: with_structured_output (Pydantic)")
    structured_llm = llm.with_structured_output(MovieReview)
    review = structured_llm.invoke("Give me a review of The Matrix (1999).")
    print(f"  Title:     {review.title}")
    print(f"  Rating:    {review.rating}")
    print(f"  Genre:     {review.genre}")
    print(f"  Summary:   {review.one_line_summary}")
    print(f"  Recommend: {review.would_recommend}")
    print(f"  Type: {type(review).__name__}")  # MovieReview!


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ PART 4 — LCEL (LangChain Expression Language)                            ┃
# ┃                                                                          ┃
# ┃ LCEL is the pipe (|) syntax for composing Runnables. It gives you:       ┃
# ┃   - Streaming for free (each step streams to the next)                   ┃
# ┃   - Batch processing with automatic parallelism                          ┃
# ┃   - Fallbacks and retries                                                ┃
# ┃   - Tracing (every step is visible in LangSmith)                         ┃
# ┃                                                                          ┃
# ┃ Key composability tools:                                                 ┃
# ┃   |            — pipe: output of left → input of right                   ┃
# ┃   RunnablePassthrough — passes input through unchanged                   ┃
# ┃   RunnableParallel    — runs multiple Runnables in parallel              ┃
# ┃   RunnableLambda      — wraps any function as a Runnable                 ┃
# ┃   .with_fallbacks()   — try alternative on failure                       ┃
# ┃   .with_retry()       — retry on transient errors                        ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough  # noqa: E402


def demo_lcel():
    """Demo: LCEL composition patterns."""
    print("\n" + "=" * 70)
    print("  PART 4: LCEL (LangChain Expression Language)")
    print("=" * 70)

    if not os.environ.get("GOOGLE_API_KEY"):
        print("  SKIPPED — set GOOGLE_API_KEY to run this demo.")
        return

    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    # --- Pattern 1: Simple chain ---
    print("\n  Pattern 1: Simple chain (prompt | model | parser)")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Translate the following to {language}. Output ONLY the translation."),
        ("human", "{text}"),
    ])
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"language": "French", "text": "Hello, how are you?"})
    print(f"  Translation: {result}")

    # --- Pattern 2: RunnableParallel (fan-out) ---
    print("\n  Pattern 2: RunnableParallel (run multiple chains at once)")
    translate_fr = (
        ChatPromptTemplate.from_messages([
            ("system", "Translate to French. Output ONLY the translation."),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )
    translate_es = (
        ChatPromptTemplate.from_messages([
            ("system", "Translate to Spanish. Output ONLY the translation."),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )

    # RunnableParallel runs both translations simultaneously
    parallel = RunnableParallel(french=translate_fr, spanish=translate_es)
    results = parallel.invoke({"text": "Good morning!"})
    print(f"  French:  {results['french']}")
    print(f"  Spanish: {results['spanish']}")

    # --- Pattern 3: RunnablePassthrough + RunnableLambda ---
    print("\n  Pattern 3: Passthrough + Lambda (data transformation)")

    def word_count(text: str) -> str:
        return f"Word count: {len(text.split())}"

    chain = (
        ChatPromptTemplate.from_messages([
            ("system", "Write a one-paragraph summary of {topic}.")
        ])
        | llm
        | StrOutputParser()
        | RunnableLambda(word_count)  # custom post-processing
    )
    result = chain.invoke({"topic": "Python"})
    print(f"  Result: {result}")

    # --- Pattern 4: Batch processing ---
    print("\n  Pattern 4: Batch processing (automatic parallelism)")
    simple_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "Answer in exactly 3 words."),
            ("human", "What is {thing}?"),
        ])
        | llm
        | StrOutputParser()
    )
    # .batch() processes multiple inputs in parallel
    results = simple_chain.batch([
        {"thing": "Python"},
        {"thing": "JavaScript"},
        {"thing": "Rust"},
    ])
    for thing, result in zip(["Python", "JavaScript", "Rust"], results):
        print(f"  {thing}: {result}")


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ PART 5 — TOOLS (Function Calling)                                        ┃
# ┃                                                                          ┃
# ┃ Tools let LLMs call Python functions. LangChain's @tool decorator        ┃
# ┃ creates a tool with name, description, and argument schema from the      ┃
# ┃ function's signature and docstring.                                      ┃
# ┃                                                                          ┃
# ┃ The flow:                                                                ┃
# ┃   1. model.bind_tools(tools) → LLM knows what tools exist               ┃
# ┃   2. LLM returns AIMessage with .tool_calls                             ┃
# ┃   3. You execute the tool and return a ToolMessage                       ┃
# ┃   4. LLM sees the result and formulates a response                       ┃
# ┃                                                                          ┃
# ┃ NOTE: This is the manual flow. For automatic tool execution, use         ┃
# ┃ LangGraph's ToolNode or create_react_agent (see langgraph_tutorial.py).  ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

from langchain_core.messages import ToolMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402


@tool
def get_population(country: str) -> str:
    """Get the approximate population of a country.

    Args:
        country: Country name (e.g. "India", "USA", "Japan").
    """
    populations = {
        "india": "1.4 billion",
        "china": "1.4 billion",
        "usa": "335 million",
        "japan": "125 million",
        "germany": "84 million",
        "brazil": "216 million",
    }
    pop = populations.get(country.lower())
    if pop:
        return f"The population of {country} is approximately {pop}."
    return f"Population data not available for {country}."


@tool
def get_capital(country: str) -> str:
    """Get the capital city of a country.

    Args:
        country: Country name (e.g. "India", "France").
    """
    capitals = {
        "india": "New Delhi",
        "china": "Beijing",
        "usa": "Washington D.C.",
        "japan": "Tokyo",
        "germany": "Berlin",
        "france": "Paris",
        "brazil": "Brasília",
    }
    cap = capitals.get(country.lower())
    if cap:
        return f"The capital of {country} is {cap}."
    return f"Capital data not available for {country}."


def demo_tools():
    """Demo: Tool/function calling with manual execution."""
    print("\n" + "=" * 70)
    print("  PART 5: Tools (Function Calling)")
    print("=" * 70)

    if not os.environ.get("GOOGLE_API_KEY"):
        print("  SKIPPED — set GOOGLE_API_KEY to run this demo.")
        return

    from langchain_google_genai import ChatGoogleGenerativeAI

    tools = [get_population, get_capital]

    # Inspect tool metadata
    print("\n  Tool metadata:")
    for t in tools:
        print(f"    {t.name}: {t.description[:60]}...")
        print(f"    Args schema: {t.args_schema.model_json_schema()['properties']}")

    # Bind tools to the model
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    # Ask a question that requires tools
    print("\n  Sending: 'What is the capital and population of Japan?'")
    response = llm_with_tools.invoke("What is the capital and population of Japan?")

    # The LLM returns tool_calls (not the final answer yet!)
    print(f"\n  Tool calls requested by LLM:")
    for tc in response.tool_calls:
        print(f"    - {tc['name']}({tc['args']})")

    # Manually execute tools and build the full message history
    messages = [
        HumanMessage(content="What is the capital and population of Japan?"),
        response,  # AIMessage with tool_calls
    ]

    # Execute each tool call and add ToolMessage results
    tool_map = {t.name: t for t in tools}
    for tc in response.tool_calls:
        tool_result = tool_map[tc["name"]].invoke(tc["args"])
        messages.append(ToolMessage(content=tool_result, tool_call_id=tc["id"]))
        print(f"    Result: {tool_result}")

    # Send everything back to the LLM for final synthesis
    final_response = llm_with_tools.invoke(messages)
    print(f"\n  Final response: {final_response.content}")


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ PART 6 — RAG (Retrieval-Augmented Generation)                            ┃
# ┃                                                                          ┃
# ┃ RAG = retrieve relevant context → inject into prompt → generate answer.  ┃
# ┃ This is the most common LLM application pattern.                         ┃
# ┃                                                                          ┃
# ┃ Components:                                                              ┃
# ┃   Document       — text + metadata                                       ┃
# ┃   TextSplitter   — splits long documents into chunks                     ┃
# ┃   Embeddings     — converts text to vectors                              ┃
# ┃   VectorStore    — stores and searches vectors                           ┃
# ┃   Retriever      — interface for fetching relevant documents             ┃
# ┃                                                                          ┃
# ┃ This demo uses a simple in-memory approach. For production, use          ┃
# ┃ FAISS, Chroma, Pinecone, or pgvector.                                   ┃
# ┃                                                                          ┃
# ┃ LIMITATION: Quality depends heavily on chunking strategy and embeddings.  ┃
# ┃ MITIGATION: Experiment with chunk sizes, overlap, and embedding models.  ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

from langchain_core.documents import Document  # noqa: E402


def demo_rag_manual():
    """Demo: Manual RAG pipeline without a vector store.

    This shows the RAG concept using simple keyword matching.
    In production, you'd use embeddings + a vector store.
    """
    print("\n" + "=" * 70)
    print("  PART 6: RAG (Retrieval-Augmented Generation)")
    print("  Using simple keyword retrieval (no vector store needed)")
    print("=" * 70)

    if not os.environ.get("GOOGLE_API_KEY"):
        print("  SKIPPED — set GOOGLE_API_KEY to run this demo.")
        return

    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    # Our "knowledge base" — in production, this would be a vector store
    documents = [
        Document(
            page_content="LangGraph is a framework for building stateful, multi-step AI agents. "
            "It models workflows as directed graphs with nodes, edges, and cycles. "
            "Key features include persistence via checkpointers and human-in-the-loop patterns.",
            metadata={"source": "langgraph_docs", "topic": "langgraph"},
        ),
        Document(
            page_content="Google ADK (Agent Development Kit) is an open-source framework for "
            "building AI agents. It supports single-agent and multi-agent architectures "
            "with LlmAgent, SequentialAgent, ParallelAgent, and LoopAgent.",
            metadata={"source": "adk_docs", "topic": "google_adk"},
        ),
        Document(
            page_content="LangChain is a framework for building LLM applications. Core concepts "
            "include Runnables, LCEL (pipe syntax), prompt templates, output parsers, "
            "tools, and chains. LangGraph and LangSmith build on top of LangChain.",
            metadata={"source": "langchain_docs", "topic": "langchain"},
        ),
    ]

    # Simple keyword retriever (in production, use embeddings)
    def simple_retrieve(query: str, docs: list[Document], top_k: int = 2) -> list[Document]:
        """Score documents by keyword overlap with the query."""
        query_words = set(query.lower().split())
        scored = []
        for doc in docs:
            doc_words = set(doc.page_content.lower().split())
            overlap = len(query_words & doc_words)
            scored.append((overlap, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]

    # The RAG chain
    query = "How does LangGraph handle state and persistence?"
    print(f"\n  Query: {query}")

    # Step 1: Retrieve relevant documents
    relevant_docs = simple_retrieve(query, documents)
    print(f"\n  Retrieved {len(relevant_docs)} documents:")
    for doc in relevant_docs:
        print(f"    - [{doc.metadata['source']}] {doc.page_content[:80]}...")

    # Step 2: Build the prompt with context
    context = "\n\n".join(doc.page_content for doc in relevant_docs)
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Answer the question based ONLY on the following context. "
            "If the context doesn't contain the answer, say so.\n\n"
            "Context:\n{context}"
        )),
        ("human", "{question}"),
    ])

    # Step 3: Generate answer
    chain = rag_prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": query})
    print(f"\n  Answer: {answer}")


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ PART 7 — ADVANCED LCEL PATTERNS                                          ┃
# ┃                                                                          ┃
# ┃ Fallbacks    — try a backup model/chain if the primary fails             ┃
# ┃ Configurable — swap components at runtime (model, temperature, etc.)     ┃
# ┃ Chaining     — complex multi-step workflows                             ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


def demo_advanced_lcel():
    """Demo: Advanced LCEL patterns — fallbacks and chaining."""
    print("\n" + "=" * 70)
    print("  PART 7: Advanced LCEL Patterns")
    print("=" * 70)

    if not os.environ.get("GOOGLE_API_KEY"):
        print("  SKIPPED — set GOOGLE_API_KEY to run this demo.")
        return

    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    # --- Pattern: Multi-step chain with data transformation ---
    print("\n  Pattern: Multi-step analysis chain")

    # Step 1: Extract key points
    extract_prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract 3 key points from the text. Return them as a numbered list."),
        ("human", "{text}"),
    ])

    # Step 2: Summarize the key points
    summarize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize these key points into one concise sentence."),
        ("human", "{key_points}"),
    ])

    # Compose the multi-step chain
    analysis_chain = (
        extract_prompt
        | llm
        | StrOutputParser()
        | (lambda key_points: {"key_points": key_points})  # reshape for next prompt
        | summarize_prompt
        | llm
        | StrOutputParser()
    )

    text = (
        "Python is a versatile programming language used in web development, "
        "data science, artificial intelligence, and automation. It has a large "
        "ecosystem of libraries including NumPy, Pandas, TensorFlow, and Flask. "
        "Python's readability and simplicity make it ideal for beginners while "
        "its power satisfies expert developers."
    )

    result = analysis_chain.invoke({"text": text})
    print(f"  Input:   {text[:80]}...")
    print(f"  Summary: {result}")

    # --- Pattern: RunnablePassthrough for injecting data ---
    print("\n  Pattern: RunnablePassthrough (inject + pass through)")

    chain = (
        RunnablePassthrough.assign(
            # Add computed fields alongside the original input
            word_count=lambda x: len(x["text"].split()),
            char_count=lambda x: len(x["text"]),
        )
        | RunnableLambda(
            lambda x: f"Text ({x['word_count']} words, {x['char_count']} chars): analyzed."
        )
    )
    result = chain.invoke({"text": "Hello world this is a test"})
    print(f"  Result: {result}")


# ===========================================================================
# MAIN — Run all demos
# ===========================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  LANGCHAIN — Zero to Hero Tutorial")
    print("  Running all demos...")
    print("=" * 70)

    if not os.environ.get("GOOGLE_API_KEY"):
        print("\n  WARNING: GOOGLE_API_KEY not set. Most demos will be skipped.")
        print("  Set it with: export GOOGLE_API_KEY='your-key'")

    demo_chat_model()
    demo_prompt_templates()
    demo_output_parsers()
    demo_lcel()
    demo_tools()
    demo_rag_manual()
    demo_advanced_lcel()

    print("\n\n" + "=" * 70)
    print("  ALL DEMOS COMPLETE!")
    print("=" * 70)
    print("""
    WHAT YOU LEARNED:
      1. ChatModel      — Uniform interface to any LLM provider
      2. Messages        — System, Human, AI, Tool message types
      3. PromptTemplate  — Parameterized prompts with variables
      4. OutputParser    — Transform LLM output to structured data
      5. LCEL            — Pipe syntax (|) for composing chains
      6. Tools           — @tool decorator + bind_tools + ToolMessage
      7. RAG             — Retrieve context → inject into prompt → generate
      8. Advanced LCEL   — Parallel, passthrough, lambda, multi-step chains

    HOW LANGCHAIN, LANGGRAPH, AND GOOGLE ADK RELATE:

      ┌─────────────────────────────────────────────────────┐
      │                                                     │
      │   LangChain = building blocks (models, prompts,     │
      │               tools, parsers, chains)               │
      │        │                                            │
      │        ▼                                            │
      │   LangGraph = orchestration (stateful agents,       │
      │              graphs, cycles, persistence)           │
      │                                                     │
      │   Google ADK = alternative to LangGraph (agents,    │
      │               tools, multi-agent, Gemini-optimized) │
      │                                                     │
      │   LangSmith = observability (tracing, eval)         │
      │                                                     │
      └─────────────────────────────────────────────────────┘

      WHEN TO USE WHAT:
        - Simple LLM calls, prompts, parsing → LangChain alone
        - Stateful agents with tool loops     → LangGraph
        - Multi-agent Gemini systems          → Google ADK
        - Any of the above + observability    → Add LangSmith

    NEXT STEPS:
      - See langgraph_tutorial.py for agent orchestration
      - See google_adk_tutorial.py for Google's agent framework
      - Explore LangSmith for tracing: https://smith.langchain.com/
      - Read: https://python.langchain.com/docs/
    """)
