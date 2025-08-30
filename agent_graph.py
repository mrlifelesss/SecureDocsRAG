# agent_graph.py
"""
Tool-using agent on top of your RAG.

The agent chooses when to call tools (your RAG tool, plus simple utilities) and
loops until it has a final answer. Designed to be imported by the Streamlit UI.

Exports:
    - agent_app: compiled LangGraph agent
    - run_agent(question, chat_history=None) -> str
    - stream_agent(question, chat_history=None) -> Iterator[str]  (optional streaming)

Requirements:
    - settings.py  (for model ids / transport)
    - agent_tools.py  (defines @tool functions: rag_answer, summarize, calculator)
"""

from __future__ import annotations
from typing_extensions import TypedDict
from typing import Annotated, List, Literal, Iterator
from uuid import uuid4

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from settings import settings as cfg
from agent_tools import rag_answer, summarize, calculator, aws_docs_search


MAX_TOOL_STEPS = 3  # hard stop to avoid infinite loops
TOOLS = [rag_answer, aws_docs_search, summarize, calculator]
TOOLS_BY_NAME: dict[str, BaseTool] = {t.name: t for t in TOOLS}

AGENT_SYSTEM = (
    "You are a tool-using security assistant with access to two knowledge bases:"
    "1. `rag_answer`: Searches a private, local knowledge base containing internal documents like the Cloud Adoption Framework (CAF) and Security Reference Architecture (SRA)."
    "2. `aws_docs_search`: Searches the official, live AWS public documentation for the most current information."
    
    "Your strategy:"
    "- For questions about internal frameworks, custom architecture, 'CAF', or 'SRA', you MUST use the `rag_answer` tool."
    "- For questions about specific AWS service features, limits, APIs, pricing, or recent updates, you MUST use the `aws_docs_search` tool to get the most accurate, up-to-date answer."
    "- If the user's query is ambiguous, start with `aws_docs_search` first."
    "- If a tool returns NO_CONTEXT or an error, do not try the other tool. Instead, ask the user a brief clarifying question."
    "- Always cite the sources provided by the tool in your final answer."
)

def trace_agent(question: str):
    """Print state updates per node to the console."""
    inputs = {"messages": [HumanMessage(content=question)], "hops": 0, "stop": False}
    dbg("TRACE START", question=question)
    for event in agent_app.stream(inputs, config={"recursion_limit": 20}, stream_mode="updates"):
        node = event.get("node")
        out = event.get("output", {})
        # Show only a compact summary
        msgs = out.get("messages", [])
        last = msgs[-1] if msgs else None
        role = type(last).__name__ if last else None
        text = (getattr(last, "content", "") or "")[:140] if last else ""
        dbg("TRACE EVENT", node=node, hops=out.get("hops"), stop=out.get("stop"), last_role=role, last_preview=text)
    dbg("TRACE END")

def _sanitize_messages(msgs: list) -> list:
    """
    Build the prompt we SEND to Gemini:
    - Keep ToolMessage objects AS-IS (they carry tool_call_id).
    - Keep any text-bearing Human/AI messages.
    - If nothing texty remains, add a minimal HumanMessage so 'contents' isn't empty.
    - Prepend a SystemMessage (NOT stored in state).
    """
    clean: list = []
    for m in msgs:
        if isinstance(m, ToolMessage):
            clean.append(m)               # do not rebuild
            continue
        text = getattr(m, "content", "")
        if isinstance(text, str):
            if text.strip():
                clean.append(m)           # keep original object
        else:
            clean.append(m)               # keep non-string content as-is

    if not any((getattr(m, "content", "") or "").strip()
               for m in clean if not isinstance(m, ToolMessage)):
        clean.append(HumanMessage(content="Hello."))

    return [SystemMessage(content=AGENT_SYSTEM)] + clean

# --- Debug logging (terminal) ---
import logging, json
LOG = logging.getLogger("agent")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
DEBUG_AGENT = True

def dbg(msg: str, **kw):
    if not DEBUG_AGENT:
        return
    try:
        extra = (" | " + json.dumps(kw, default=str)) if kw else ""
    except Exception:
        extra = ""
    LOG.info(msg + extra)

# ---------------------------- Agent LLM (Gemini) ---------------------------- #
# Note: binding tools enables function/tool calling; temperature=0 for stability.
agent_llm = ChatGoogleGenerativeAI(
    model=cfg.GEMINI_MODEL,
    transport=cfg.GEMINI_TRANSPORT,
    temperature=0,
).bind_tools([rag_answer,aws_docs_search, summarize, calculator])


# ------------------------------- Agent State -------------------------------- #
class AgentState(TypedDict, total=False):
    """
    Conversation state for the agent graph.

    Keys:
        messages: list[BaseMessage]
            The rolling conversation. We only ever append to it.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    hops: int
    stop: bool


# ------------------------------ Graph Nodes --------------------------------- #
def agent_node(state: AgentState) -> AgentState:
    """
    Call the tool-enabled LLM unless we've hit the hop cap.
    If cap is reached we return a lightweight final message to end the turn.
    """
    hops = state.get("hops", 0)
    stop = state.get("stop", False)

    last = state["messages"][-1]
    if isinstance(last, ToolMessage):
        txt = (last.content or "")
        if isinstance(txt, str) and (txt.startswith("ERROR:") or txt.startswith("NO_CONTEXT:")):
            ai = AIMessage(
                content=("I couldn’t find enough grounded evidence to answer. "
                         "Please narrow the topic (service/scope/keyword)."))
            return {"messages": state["messages"] + [ai], "hops": hops, "stop": stop}

    # existing hop-cap / stop guard
    if hops >= MAX_TOOL_STEPS or stop:
        ai = AIMessage(content=("I don’t have enough specific context to proceed. "
                                "Please narrow the topic (e.g., IAM/S3/KMS, account/org scope, or a keyword)."))
        return {"messages": state["messages"] + [ai], "hops": hops, "stop": stop}

    prompt_msgs = _sanitize_messages(state["messages"])
    ai = agent_llm.invoke(prompt_msgs)
    return {"messages": state["messages"] + [ai], "hops": hops, "stop": stop}


# Tool executor node: runs any tool calls produced by the last AI message.
def _norm_args(args):
    """Normalize various providers' tool call arg shapes to kwargs for our tools."""
    import json
    if args is None:
        return {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            return {"question": args}
    if not isinstance(args, dict):
        return {"question": str(args)}

    # unify to 'question' for rag_answer
    if "question" not in args:
        for k in ("input", "query", "prompt", "text"):
            if k in args and isinstance(args[k], str):
                args["question"] = args[k]
                break
    return args

def tools_runner(state: AgentState) -> AgentState:
    last = state["messages"][-1]
    calls = getattr(last, "tool_calls", None) or getattr(last, "additional_kwargs", {}).get("tool_calls", [])
    if not calls:
        dbg("TOOLS: no calls", hops=state.get("hops", 0))
        return {"messages": state["messages"], "hops": state.get("hops", 0), "stop": state.get("stop", False)}

    out_msgs, stop = [], state.get("stop", False)
    for c in calls:
        name = c.get("name") or (c.get("function") or {}).get("name")
        raw_args = c.get("args") or (c.get("function") or {}).get("arguments") or c.get("input")
        kwargs = _norm_args(raw_args)
        call_id = c.get("id") or c.get("tool_call_id") or f"call_{uuid4().hex[:8]}"
        dbg("TOOLS: exec", name=name, kwargs_preview=str(kwargs)[:160])

        tool = TOOLS_BY_NAME.get(name or "")
        if not tool:
            content = f"ERROR: unknown tool '{name}'"
        else:
            try:
                result = tool.invoke(kwargs)
                content = str(result)
            except Exception as e:
                content = f"ERROR: {e}"

        # Decide whether to stop retrying
        if content.startswith("NO_CONTEXT:") or content.startswith("ERROR:"):
            stop = True
        if content.startswith(("NO_CONTEXT:", "ERROR:", "Confidence: Low")):
            stop = True
        dbg("TOOLS: result", name=name, stop=stop, content_preview=content[:160])
        out_msgs.append(ToolMessage(content=content, name=name or "unknown", tool_call_id=call_id))

    return {"messages": state["messages"] + out_msgs, "hops": state.get("hops", 0) + 1, "stop": stop}

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    Router: if the last AI message contains tool calls, run tools; otherwise end.
    LangChain's AIMessage exposes tool calls via `tool_calls` (preferred) or
    `additional_kwargs["tool_calls"]` depending on backend versions.
    """
    if state.get("stop", False):
        dbg("ROUTER: stop flag set → END", hops=state.get("hops"))
        return "end"
    hops = state.get("hops", 0)
    if hops >= MAX_TOOL_STEPS:
        dbg("ROUTER: hop cap reached → END", hops=hops)
        return "end"
    last = state["messages"][-1]
    calls = getattr(last, "tool_calls", None) or getattr(last, "additional_kwargs", {}).get("tool_calls", [])
    dbg("ROUTER:", hops=hops, n_calls=len(calls))
    return "tools" if calls else "end"


# ----------------------------- Build the graph ------------------------------ #
_graph = StateGraph(AgentState)
_graph.add_node("agent", agent_node)
_graph.add_node("tools", tools_runner)

_graph.add_edge(START, "agent")
_graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
_graph.add_edge("tools", "agent")

agent_app = _graph.compile()


# ------------------------------- Public API -------------------------------- #
def run_agent(question: str, chat_history: List[BaseMessage] | None = None) -> str:
    messages = (chat_history or []) + [HumanMessage(content=question)]
    out = agent_app.invoke(
        {"messages": messages, "hops": 0, "stop": False},
        config={"recursion_limit": 20},   # headroom while testing
    )
    final = out["messages"][-1]
    return getattr(final, "content", "") or ""


def stream_agent(question: str, chat_history: List[BaseMessage] | None = None) -> Iterator[str]:
    """Yield the final answer once (robust against empty/invalid message stacks)."""
    try:
        answer = run_agent(question, chat_history)
        yield answer
    except Exception as e:
        # Graceful fallback so Streamlit doesn't crash
        yield f"Sorry—agent failed to stream. Error: {e}"