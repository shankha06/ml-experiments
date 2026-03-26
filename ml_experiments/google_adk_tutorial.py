"""
=============================================================================
Google ADK (Agent Development Kit) — Zero to Hero Tutorial
=============================================================================

WHAT IS GOOGLE ADK?
    Google ADK is an open-source, code-first Python framework (Apache 2.0) for
    building, evaluating, and deploying AI-powered agents. Developed by Google,
    it is optimized for Gemini models but is model-agnostic.

    Think of it as: "Flask/FastAPI but for AI agents" — you define agents in
    Python, wire them together, and ADK handles the LLM orchestration, tool
    calling, session management, and event streaming.

ARCHITECTURE OVERVIEW:
    ┌─────────────────────────────────────────────────────────┐
    │                       Runner                            │
    │  (orchestrates execution, manages sessions, streams     │
    │   events back to the caller)                            │
    │                                                         │
    │   ┌───────────────┐    ┌──────────────────────┐         │
    │   │   Session      │    │   Agent               │        │
    │   │  (conversation │    │  ┌─────────────────┐  │        │
    │   │   thread with  │◄──►│  │  LLM (Gemini)   │  │        │
    │   │   events &     │    │  └────────┬────────┘  │        │
    │   │   state)       │    │           │           │        │
    │   └───────────────┘    │  ┌────────▼────────┐  │        │
    │                        │  │  Tools (Python   │  │        │
    │                        │  │  functions)      │  │        │
    │                        │  └─────────────────┘  │        │
    │                        │  ┌─────────────────┐  │        │
    │                        │  │  Sub-Agents      │  │        │
    │                        │  └─────────────────┘  │        │
    │                        └──────────────────────┘         │
    └─────────────────────────────────────────────────────────┘

KEY CONCEPTS:
    Agent       — An LLM-powered entity with instructions, tools, and sub-agents.
                  Three types: LlmAgent (dynamic), WorkflowAgent (deterministic),
                  CustomAgent (extend BaseAgent).
    Tool        — A Python function the agent can call. ADK auto-reads the
                  signature, type hints, and docstring to generate the schema.
    Runner      — Orchestrates execution: retrieves sessions, sends messages to
                  agents, streams events back. InMemoryRunner for dev, Runner
                  with custom services for production.
    Session     — A conversation thread containing Events (messages/actions) and
                  State (key-value data). Managed by a SessionService.
    Event       — A single unit of activity: user message, agent response, tool
                  call, tool result, etc.

CAPABILITIES:
    - Single-agent with tool calling
    - Multi-agent hierarchies (coordinator → specialists)
    - Deterministic workflow agents (Sequential, Parallel, Loop)
    - Structured JSON output with output_schema
    - Session state for cross-turn data sharing
    - Callbacks/hooks for intercepting events
    - Built-in tools: google_search, code_execution
    - MCP (Model Context Protocol) tool integration
    - CLI: `adk run`, `adk web` (dev UI), `adk api_server`

LIMITATIONS & MITIGATIONS:
    1. Gemini-first: While model-agnostic, best tested with Gemini. Non-Gemini
       models may have quirks with tool calling formats.
       → Mitigation: Stick to Gemini for production; test thoroughly with others.
    2. InMemoryRunner loses data on restart.
       → Mitigation: Use DatabaseSessionService or custom persistence for prod.
    3. No built-in rate limiting or retry logic.
       → Mitigation: Wrap tool functions with tenacity or custom retry decorators.
    4. Sub-agent routing depends on LLM interpreting descriptions correctly.
       → Mitigation: Write very clear, distinct descriptions. Test edge cases.
    5. Async-first internally; synchronous wrapper (runner.run) exists but
       may not suit all use cases.
       → Mitigation: Use async runner.run_async() in async applications.
    6. Limited observability out of the box.
       → Mitigation: Use callbacks or integrate with OpenTelemetry.

SETUP:
    pip install google-adk   # or: uv add google-adk
    export GOOGLE_API_KEY='your-key'  # from https://aistudio.google.com/apikey

RUN:
    GOOGLE_API_KEY=your-key uv run python ml_experiments/google_adk_tutorial.py

REFERENCE:
    Docs:   https://google.github.io/adk-docs/
    GitHub: https://github.com/google/adk-python
    PyPI:   https://pypi.org/project/google-adk/
=============================================================================
"""

from __future__ import annotations

import datetime
import os
import sys
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Guard: ensure API key is set before importing ADK (which may call Google APIs)
# ---------------------------------------------------------------------------
if not os.environ.get("GOOGLE_API_KEY"):
    print("ERROR: Set GOOGLE_API_KEY environment variable first.")
    print("  export GOOGLE_API_KEY='your-key-here'")
    print("  Get one at: https://aistudio.google.com/apikey")
    sys.exit(1)

from google.adk.agents import Agent, LoopAgent, ParallelAgent, SequentialAgent  # noqa: E402
from google.adk.runners import InMemoryRunner  # noqa: E402
from google.genai import types as genai_types  # noqa: E402

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ PART 1 — TOOLS                                                          ┃
# ┃                                                                          ┃
# ┃ Tools are the #1 way agents interact with the outside world.             ┃
# ┃ In ADK, a tool is simply a Python function. ADK auto-wraps it into a     ┃
# ┃ FunctionTool by inspecting:                                              ┃
# ┃   - Function name  → tool name                                          ┃
# ┃   - Docstring      → tool description (shown to the LLM)                ┃
# ┃   - Type hints     → parameter types                                    ┃
# ┃   - Args section   → parameter descriptions                             ┃
# ┃                                                                          ┃
# ┃ BEST PRACTICES:                                                          ┃
# ┃   ✓ Always return a dict with a "status" key ("success" or "error")      ┃
# ┃   ✓ Use clear docstrings — the LLM reads them to decide when to call    ┃
# ┃   ✓ Keep parameter types simple (str, int, float, bool, list, dict)      ┃
# ┃   ✓ Validate inputs inside the function                                  ┃
# ┃   ✗ Don't use *args, **kwargs (ADK can't generate schema for them)       ┃
# ┃   ✗ Don't return non-serializable objects (use dicts/strings)            ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


def get_weather(city: str) -> dict:
    """Get the current weather for a city.

    This tool looks up weather data for major cities. Use it when the user
    asks about weather, temperature, or conditions in a specific location.

    Args:
        city: Name of the city (e.g. "London", "Tokyo", "New York").

    Returns:
        dict with keys: status, city, temp_c, condition.
        On failure: status="error" with a message.
    """
    weather_db = {
        "london": {"temp_c": 14, "condition": "cloudy", "humidity": 78},
        "tokyo": {"temp_c": 26, "condition": "sunny", "humidity": 45},
        "new york": {"temp_c": 18, "condition": "rainy", "humidity": 82},
        "paris": {"temp_c": 16, "condition": "partly cloudy", "humidity": 65},
        "sydney": {"temp_c": 22, "condition": "clear", "humidity": 55},
    }
    info = weather_db.get(city.lower())
    if info is None:
        return {"status": "error", "message": f"No weather data for '{city}'. Try: London, Tokyo, New York, Paris, Sydney."}
    return {"status": "success", "city": city, **info}


def convert_temperature(temp_c: float, to_unit: str = "fahrenheit") -> dict:
    """Convert a Celsius temperature to Fahrenheit or Kelvin.

    Args:
        temp_c: Temperature in Celsius.
        to_unit: Target unit — "fahrenheit" or "kelvin" (default: "fahrenheit").

    Returns:
        dict with status, value, and unit.
    """
    if to_unit.lower() == "fahrenheit":
        return {"status": "success", "value": round(temp_c * 9 / 5 + 32, 1), "unit": "°F"}
    elif to_unit.lower() == "kelvin":
        return {"status": "success", "value": round(temp_c + 273.15, 2), "unit": "K"}
    return {"status": "error", "message": f"Unknown unit '{to_unit}'. Use 'fahrenheit' or 'kelvin'."}


def get_current_time(city: str) -> dict:
    """Get the current time in a city.

    Args:
        city: City name (supports: New York, London, Tokyo, Paris, Sydney).

    Returns:
        dict with status, city, time, and timezone.
    """
    tz_map = {
        "new york": "America/New_York",
        "london": "Europe/London",
        "tokyo": "Asia/Tokyo",
        "paris": "Europe/Paris",
        "sydney": "Australia/Sydney",
    }
    tz_name = tz_map.get(city.lower())
    if tz_name is None:
        return {"status": "error", "message": f"Unknown city '{city}'. Try: New York, London, Tokyo, Paris, Sydney."}
    now = datetime.datetime.now(ZoneInfo(tz_name))
    return {"status": "success", "city": city, "time": now.strftime("%I:%M %p"), "timezone": tz_name}


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ PART 2 — CREATING AGENTS                                                ┃
# ┃                                                                          ┃
# ┃ Agent (aliased from LlmAgent) is the core building block.               ┃
# ┃                                                                          ┃
# ┃ Constructor parameters:                                                  ┃
# ┃   name         (str, required)  — unique identifier                      ┃
# ┃   model        (str, required)  — LLM model id                          ┃
# ┃   instruction  (str)            — system prompt                          ┃
# ┃   description  (str)            — used by parent agents for routing      ┃
# ┃   tools        (list)           — Python functions or BaseTool instances  ┃
# ┃   sub_agents   (list)           — child agents for multi-agent setups    ┃
# ┃   output_key   (str)            — store response in session state        ┃
# ┃   output_schema(type)           — enforce structured JSON output         ┃
# ┃   generate_content_config       — temperature, max_tokens, safety, etc.  ┃
# ┃   planner      (Planner)        — for multi-step planning (ReAct, etc.)  ┃
# ┃                                                                          ┃
# ┃ TIPS:                                                                    ┃
# ┃   - instruction is the most important parameter. Be specific and clear.  ┃
# ┃   - description matters for multi-agent routing — make it distinctive.   ┃
# ┃   - Use output_key to pass data between agents in a SequentialAgent.     ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# --- Agent 1: Weather assistant with multiple tools ---
weather_agent = Agent(
    name="weather_assistant",
    model="gemini-2.5-flash",
    instruction=(
        "You are a helpful weather assistant. "
        "When the user asks about weather, use the get_weather tool. "
        "If they want temperature conversion, use convert_temperature. "
        "If they ask about the time, use get_current_time. "
        "Always be concise and friendly. "
        "If a tool returns an error, relay the error message helpfully."
    ),
    description="Answers weather and time questions using tools.",
    tools=[get_weather, convert_temperature, get_current_time],
)

# --- Agent 2: A simple agent with no tools ---
joke_agent = Agent(
    name="joke_teller",
    model="gemini-2.5-flash",
    instruction=(
        "You are a witty comedian. Tell short, family-friendly jokes. "
        "Keep responses to 1-3 lines. If asked for a specific type of joke, "
        "tailor it accordingly."
    ),
    description="Tells jokes and handles humor/entertainment requests.",
)


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ PART 3 — STRUCTURED OUTPUT                                              ┃
# ┃                                                                          ┃
# ┃ Use output_schema to force the agent to return structured JSON.          ┃
# ┃ The schema can be a Pydantic model or a plain dict schema.               ┃
# ┃                                                                          ┃
# ┃ WHY: Guarantees parseable output for downstream systems.                 ┃
# ┃ LIMITATION: The LLM may struggle with very complex schemas.              ┃
# ┃ MITIGATION: Keep schemas flat and simple; test with real queries.        ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

from pydantic import BaseModel, Field  # noqa: E402


class TravelRecommendation(BaseModel):
    """Structured output for travel recommendations."""

    city: str = Field(description="The recommended city")
    reason: str = Field(description="Why this city is recommended")
    best_season: str = Field(description="Best season to visit")
    budget_level: str = Field(description="Budget level: low, medium, or high")


travel_agent = Agent(
    name="travel_recommender",
    model="gemini-2.5-flash",
    instruction=(
        "You recommend travel destinations. When asked, suggest a city "
        "and provide structured details. Be specific and helpful."
    ),
    description="Recommends travel destinations with structured output.",
    output_schema=TravelRecommendation,
    # NOTE: When output_schema is set, the agent's final response will be
    # valid JSON matching the schema instead of free-form text.
)


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ PART 4 — RUNNER & SESSIONS                                              ┃
# ┃                                                                          ┃
# ┃ The Runner is what actually executes agents:                             ┃
# ┃   1. Retrieves (or creates) a Session for the given user_id+session_id   ┃
# ┃   2. Appends the user's message as an Event                             ┃
# ┃   3. Sends the conversation to the Agent                                 ┃
# ┃   4. Streams back Events (tool calls, tool results, agent responses)     ┃
# ┃                                                                          ┃
# ┃ InMemoryRunner = Runner + InMemorySessionService + InMemoryArtifactService┃
# ┃ Good for development. Data lost on restart.                              ┃
# ┃                                                                          ┃
# ┃ For production: use Runner with DatabaseSessionService or custom impl.   ┃
# ┃                                                                          ┃
# ┃ SESSIONS:                                                                ┃
# ┃   - Same user_id + session_id = same conversation (messages accumulate)  ┃
# ┃   - Different session_id = fresh conversation                            ┃
# ┃   - session.state = key-value store for cross-turn data                  ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

runner = InMemoryRunner(agent=weather_agent)


def ask(
    question: str,
    agent_runner: InMemoryRunner | None = None,
    user_id: str = "user_1",
    session_id: str = "session_1",
) -> None:
    """Send a question to an agent and print the streamed response.

    This helper demonstrates the core execution loop:
      1. Wrap the question in a Content object (ADK's message format)
      2. Call runner.run() which yields Events
      3. Filter for events that have text content and print them

    Args:
        question: The user's question.
        agent_runner: Which runner to use (defaults to the weather_agent runner).
        user_id: User identifier for session scoping.
        session_id: Session identifier (same = same conversation).
    """
    if agent_runner is None:
        agent_runner = runner

    print(f"\n{'='*60}")
    print(f"USER: {question}")
    print("-" * 60)

    # ADK uses google.genai.types for message format (shared with Gemini SDK)
    message = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=question)],
    )

    # runner.run() is a generator that yields Event objects
    for event in agent_runner.run(user_id=user_id, session_id=session_id, new_message=message):
        # Each event has: .author (str), .content (Content), .actions, etc.
        # We check for text content; tool_call events have non-text parts.
        if event.content and event.content.parts:
            text = event.content.parts[0].text
            if text:
                print(f"[{event.author}]: {text}")


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ PART 5 — MULTI-AGENT SYSTEMS                                            ┃
# ┃                                                                          ┃
# ┃ ADK supports hierarchical agent architectures:                           ┃
# ┃                                                                          ┃
# ┃   coordinator (parent)                                                   ┃
# ┃   ├── weather_assistant (sub-agent with tools)                           ┃
# ┃   ├── joke_teller (sub-agent, no tools)                                  ┃
# ┃   └── travel_recommender (sub-agent, structured output)                  ┃
# ┃                                                                          ┃
# ┃ HOW ROUTING WORKS:                                                       ┃
# ┃   The parent agent sees the `description` of each sub-agent and uses     ┃
# ┃   it (via the LLM) to decide which sub-agent should handle the request.  ┃
# ┃   This is LLM-based routing, not rule-based.                             ┃
# ┃                                                                          ┃
# ┃ LIMITATION: Routing is non-deterministic. The LLM might misroute.        ┃
# ┃ MITIGATION: Write clear, non-overlapping descriptions. Test edge cases.  ┃
# ┃             For deterministic routing, use WorkflowAgents instead.        ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

coordinator = Agent(
    name="coordinator",
    model="gemini-2.5-flash",
    instruction=(
        "You are a helpful coordinator. Based on the user's request:\n"
        "- Route weather/time questions to weather_assistant\n"
        "- Route joke/humor requests to joke_teller\n"
        "- Route travel/destination questions to travel_recommender\n"
        "- For anything else, answer directly and briefly."
    ),
    description="Routes user requests to the appropriate specialist agent.",
    sub_agents=[weather_agent, joke_agent, travel_agent],
)


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ PART 6 — WORKFLOW AGENTS (Deterministic Orchestration)                   ┃
# ┃                                                                          ┃
# ┃ Unlike LLM agents, workflow agents follow FIXED execution patterns.      ┃
# ┃ They don't use an LLM for routing — the flow is hardcoded.               ┃
# ┃                                                                          ┃
# ┃ Types:                                                                   ┃
# ┃   SequentialAgent — runs sub-agents one after another, in order          ┃
# ┃   ParallelAgent   — runs sub-agents concurrently, merges results         ┃
# ┃   LoopAgent       — runs sub-agents repeatedly until a condition is met  ┃
# ┃                                                                          ┃
# ┃ USE CASES:                                                               ┃
# ┃   - ETL pipelines: extract → transform → load                           ┃
# ┃   - Parallel data fetching: get weather + get news simultaneously        ┃
# ┃   - Iterative refinement: write → review → revise (loop until good)      ┃
# ┃                                                                          ┃
# ┃ KEY DIFFERENCE from LLM routing:                                         ┃
# ┃   LLM agents:      "LLM, which sub-agent should handle this?"           ┃
# ┃   Workflow agents:  "Run A, then B, then C. No LLM decision needed."    ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# --- SequentialAgent: weather → travel advice ---
# The travel_advisor reads the conversation (which now has weather info)
# and gives advice based on it.
travel_advisor_agent = Agent(
    name="travel_advisor",
    model="gemini-2.5-flash",
    instruction=(
        "Based on the weather information in the conversation so far, "
        "give a short travel tip: what to wear, what to pack, and one "
        "activity suggestion. Keep it to 2-3 sentences."
    ),
    description="Gives travel advice based on current weather context.",
)

weather_then_advice = SequentialAgent(
    name="weather_pipeline",
    description="Gets weather info, then gives travel advice based on it.",
    sub_agents=[weather_agent, travel_advisor_agent],
    # Sub-agents run in ORDER: weather_agent first, then travel_advisor_agent.
    # The conversation context flows through, so the second agent sees what
    # the first agent said.
)

# --- ParallelAgent: fetch multiple things at once ---
time_agent = Agent(
    name="time_checker",
    model="gemini-2.5-flash",
    instruction="Tell the user what time it is in Tokyo using the get_current_time tool.",
    description="Gets current time.",
    tools=[get_current_time],
)

parallel_fetcher = ParallelAgent(
    name="parallel_info",
    description="Gets weather and time simultaneously.",
    sub_agents=[weather_agent, time_agent],
    # Both agents run concurrently! Results are merged into the conversation.
)

# --- LoopAgent: iterative refinement ---
# LoopAgent runs its sub-agents repeatedly. The loop ends when:
#   - A sub-agent sets escalate=True in its response, OR
#   - max_iterations is reached
writer_agent = Agent(
    name="writer",
    model="gemini-2.5-flash",
    instruction=(
        "Write or revise a short haiku based on the conversation. "
        "If a reviewer has given feedback, incorporate it."
    ),
    description="Writes/revises a haiku.",
)

reviewer_agent = Agent(
    name="reviewer",
    model="gemini-2.5-flash",
    instruction=(
        "Review the haiku written by the writer. "
        "If it's good (proper 5-7-5 syllable structure), say 'APPROVED' and nothing else. "
        "If not, give specific feedback for improvement."
    ),
    description="Reviews haiku quality.",
)

haiku_loop = LoopAgent(
    name="haiku_refiner",
    description="Iteratively writes and refines a haiku.",
    sub_agents=[writer_agent, reviewer_agent],
    max_iterations=3,  # safety limit to prevent infinite loops
)


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ PART 7 — SESSION STATE (Cross-Turn Data Sharing)                         ┃
# ┃                                                                          ┃
# ┃ Agents can read/write to session.state — a key-value store that          ┃
# ┃ persists across turns within the same session.                           ┃
# ┃                                                                          ┃
# ┃ Use output_key to automatically save an agent's response to state.       ┃
# ┃ Other agents in a SequentialAgent can then read it.                      ┃
# ┃                                                                          ┃
# ┃ EXAMPLE:                                                                 ┃
# ┃   Agent A (output_key="weather_data") → saves response to state          ┃
# ┃   Agent B (instruction uses {weather_data}) → reads from state           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

data_collector = Agent(
    name="data_collector",
    model="gemini-2.5-flash",
    instruction="Get the weather in London using the get_weather tool. Report it concisely.",
    tools=[get_weather],
    output_key="collected_weather",  # saves this agent's response to session state
)

summarizer = Agent(
    name="summarizer",
    model="gemini-2.5-flash",
    instruction=(
        "The collected weather data is: {collected_weather}\n"  # reads from state!
        "Summarize it in one sentence and suggest if it's a good day for a picnic."
    ),
)

stateful_pipeline = SequentialAgent(
    name="stateful_pipeline",
    description="Collects weather data, then summarizes it using state.",
    sub_agents=[data_collector, summarizer],
)


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ PART 8 — CALLBACKS (Event Interception)                                  ┃
# ┃                                                                          ┃
# ┃ ADK supports before/after callbacks on tool calls and agent responses.   ┃
# ┃ Useful for: logging, validation, modifying behavior, guardrails.         ┃
# ┃                                                                          ┃
# ┃ Callback types:                                                          ┃
# ┃   before_tool_callback  — called before a tool executes                  ┃
# ┃   after_tool_callback   — called after a tool returns                    ┃
# ┃   before_model_callback — called before the LLM is invoked              ┃
# ┃   after_model_callback  — called after the LLM responds                 ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


def log_tool_call(tool, args, tool_context):
    """A before_tool_callback that logs every tool call.

    Args:
        tool: The tool being called.
        args: The arguments passed to the tool.
        tool_context: Context with session info, state, etc.

    Returns:
        None to proceed normally, or a dict to short-circuit (skip the tool).
    """
    print(f"  [CALLBACK] Tool '{tool.name}' called with args: {args}")
    return None  # return None = proceed with the tool call


logged_agent = Agent(
    name="logged_weather",
    model="gemini-2.5-flash",
    instruction="Answer weather questions using the get_weather tool.",
    tools=[get_weather],
    before_tool_callback=log_tool_call,
)


# ===========================================================================
# MAIN — Run all demos
# ===========================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  GOOGLE ADK — Zero to Hero Tutorial")
    print("  Running all demos...")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Demo 1: Single agent with tools
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("  DEMO 1: Single Agent with Tools")
    print("  Shows: tool calling, session memory (same session_id)")
    print("=" * 70)

    ask("What's the weather like in Tokyo?")
    # Second question uses same session_id, so agent remembers the context
    ask("Can you convert that temperature to Fahrenheit?")
    ask("What time is it there?")

    # -----------------------------------------------------------------------
    # Demo 2: Multi-agent routing
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("  DEMO 2: Multi-Agent Routing (Coordinator)")
    print("  Shows: LLM-based routing to specialist sub-agents")
    print("=" * 70)

    multi_runner = InMemoryRunner(agent=coordinator)
    for q in [
        "Tell me a joke about programming",
        "What's the weather in Paris?",
        "Recommend a travel destination for winter",
    ]:
        ask(q, agent_runner=multi_runner, session_id="multi_demo")

    # -----------------------------------------------------------------------
    # Demo 3: Sequential workflow
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("  DEMO 3: SequentialAgent (Weather → Travel Advice)")
    print("  Shows: deterministic pipeline, context flowing between agents")
    print("=" * 70)

    seq_runner = InMemoryRunner(agent=weather_then_advice)
    ask("I'm planning to visit London tomorrow.", agent_runner=seq_runner, session_id="seq_demo")

    # -----------------------------------------------------------------------
    # Demo 4: Session state
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("  DEMO 4: Session State (output_key)")
    print("  Shows: agent A saves data to state, agent B reads it")
    print("=" * 70)

    state_runner = InMemoryRunner(agent=stateful_pipeline)
    ask("Check the weather for me.", agent_runner=state_runner, session_id="state_demo")

    # -----------------------------------------------------------------------
    # Demo 5: Callback logging
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("  DEMO 5: Callbacks (Tool Call Logging)")
    print("  Shows: before_tool_callback intercepting tool calls")
    print("=" * 70)

    cb_runner = InMemoryRunner(agent=logged_agent)
    ask("What's the weather in Sydney?", agent_runner=cb_runner, session_id="cb_demo")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("  ALL DEMOS COMPLETE!")
    print("=" * 70)
    print("""
    WHAT YOU LEARNED:
      1. Tools       — Python functions auto-wrapped as agent tools
      2. Agents      — LLM-powered entities with instructions & tools
      3. Runner      — Orchestrates execution and streams events
      4. Sessions    — Conversation threads with state
      5. Multi-Agent — Coordinator routes to specialists via descriptions
      6. Workflows   — Sequential, Parallel, Loop agents for deterministic flows
      7. State       — output_key to pass data between agents
      8. Callbacks   — Intercept tool calls for logging/validation
      9. Structured  — output_schema for guaranteed JSON responses

    NEXT STEPS:
      - Try `adk web` for the built-in dev UI
      - Explore built-in tools: google_search, code_execution
      - Add MCP tools for external integrations
      - Use DatabaseSessionService for persistence
      - Read: https://google.github.io/adk-docs/
    """)
