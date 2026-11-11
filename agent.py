"""
Assignment 3: the agent (`main.py ask "question"` runs `python agent.py "question"`).

LangChain `create_agent` on Gemini, with the tools of server.py over MCP and
AGENTS.md as the system prompt. Three guardrails, all run by the harness, so
the model cannot skip them:

  approval      add_document runs only after the person at the keyboard types y
  call limit    at most 8 model calls per question (stops runaway loops)
  grounding     every citation in the answer must name a passage that search
                returned in this run, and an answer that is not a refusal must
                cite something (except a reply to a request to add a document);
                otherwise the model gets one retry, and if it fails again the
                answer is replaced by a refusal

    uv run python agent.py "Can I fly my C1 drone over people?"

evaluate.py imports load_mcp_tools, build_agent and ask.
"""

import asyncio
import re
import sys
import time
import warnings
from contextlib import AsyncExitStack
from pathlib import Path

from fastmcp.client.transports import PythonStdioTransport
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, wrap_tool_call
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from api_key import load_api_key

with warnings.catch_warnings():   # langchain.mcp is in beta and warns on import
    warnings.simplefilter("ignore")
    from langchain.mcp import MCPAdapter

ROOT = Path(__file__).resolve().parent
MODEL = "gemini-3.5-flash-lite"   # fast, cheap, generous free-tier quota
SERVER = ROOT / "server.py"
NEEDS_APPROVAL = {"add_document"}  # changes the corpus and spends embedding quota
MAX_MODEL_CALLS = 8
REFUSAL = "The documents do not cover this."
CITATION = re.compile(r"\[([a-z0-9][a-z0-9-]*), ([^\[\]]+?)\]")
RESULT_HEADER = re.compile(r"^\d+\. \[([a-z0-9][a-z0-9-]*), ([^\[\]]+)\]$")


# 1. Tools: start server.py over stdio and wrap its tools as LangChain tools.
#    MCP starts servers with a minimal environment, so the API key is passed explicitly.
async def load_mcp_tools(stack: AsyncExitStack):
    transport = PythonStdioTransport(
        SERVER, env={"GEMINI_API_KEY": load_api_key()}, cwd=str(ROOT), keep_alive=False
    )
    adapter = await stack.enter_async_context(MCPAdapter(transport))
    return await adapter.list_tools()


# 2. Guardrail: approval before add_document. `approve` decides; the default asks in the
#    terminal and refuses when there is no terminal to ask (check.py, GitHub Actions, evaluate.py).
def ask_in_terminal(call: dict) -> bool:
    if not sys.stdin.isatty():
        return False
    answer = input(f"\n  Allow {call['name']}({call['args']})? [y/N] ")
    return answer.strip().lower() == "y"


def approval_guardrail(approve):
    @wrap_tool_call
    async def ask_before_changing_the_corpus(request, handler):
        call = request.tool_call
        if call["name"] in NEEDS_APPROVAL and not approve(call):
            return ToolMessage(
                content="The user refused this tool call. Do not try to add the document another way.",
                tool_call_id=call["id"],
                status="error",
            )
        return await handler(request)

    return ask_before_changing_the_corpus


# 3. The agent: model + tools + AGENTS.md + guardrails. create_agent runs the tool loop.
def build_agent(tools, approve=ask_in_terminal):
    return create_agent(
        model=ChatGoogleGenerativeAI(model=MODEL),
        tools=tools,
        system_prompt=(ROOT / "AGENTS.md").read_text(encoding="utf-8"),
        middleware=[
            approval_guardrail(approve),
            ModelCallLimitMiddleware(run_limit=MAX_MODEL_CALLS),
        ],
    )


def tool_trace(messages) -> list[str]:
    """Every tool call and its result, in order: what the harness saw happen."""
    lines = []
    for message in messages:
        if isinstance(message, AIMessage):
            lines += [f"call {c['name']}({c['args']})" for c in message.tool_calls]
        elif isinstance(message, ToolMessage):
            lines.append(f"result {message.text}")
    return lines


def retrieved_passages(messages) -> dict[tuple[str, str], str]:
    """{(source, section): header line} for every passage that search returned."""
    passages = {}
    for message in messages:
        if isinstance(message, ToolMessage) and message.name == "search":
            lines = message.text.splitlines()
            for i, line in enumerate(lines):
                header = RESULT_HEADER.match(line.strip())
                if header:
                    detail = lines[i + 1].strip() if i + 1 < len(lines) else ""
                    passages[(header[1], header[2])] = detail
    return passages


def citations(answer: str) -> list[tuple[str, str]]:
    return list(dict.fromkeys((m[1], m[2].strip()) for m in CITATION.finditer(answer)))


def is_refusal(answer: str) -> bool:
    """AGENTS.md rule 5: a full refusal contains this exact sentence (it may add one cited sentence)."""
    return REFUSAL in answer


# 4. Guardrail: grounding. Checked in code after the model has answered.
def grounding_problem(answer: str, passages: dict, action_request: bool = False) -> str | None:
    """Why the answer fails the check, or None if it passes.

    A reply to a request to add a document reports what the tool did, not facts
    from the corpus, so it need not cite; any citation it does contain is still checked.
    """
    cited = citations(answer)
    invented = [f"[{s}, {sec}]" for s, sec in cited if (s, sec) not in passages]
    if invented:
        return f"it cites {', '.join(invented)}, which search did not return in this conversation"
    if not cited and not is_refusal(answer) and not action_request:
        return "it states facts but cites no passage"
    return None


def tokens_used(messages) -> dict[str, int]:
    used = {"input": 0, "output": 0}
    for message in messages:
        usage = getattr(message, "usage_metadata", None) if isinstance(message, AIMessage) else None
        if usage:
            used["input"] += usage.get("input_tokens", 0)
            used["output"] += usage.get("output_tokens", 0)
    return used


async def ask(agent, question: str) -> dict:
    """Run one question. Returns the answer, the tool trace, what was retrieved, cost and guardrail events."""
    started = time.time()
    result = await agent.ainvoke({"messages": [HumanMessage(question)]})
    messages = result["messages"]
    answer = messages[-1].text.strip()
    guardrail = None
    action_request = any(isinstance(m, AIMessage) and any(c["name"] in NEEDS_APPROVAL for c in m.tool_calls)
                         for m in messages)

    problem = grounding_problem(answer, retrieved_passages(messages), action_request)
    if problem:
        guardrail = {"fired": problem, "outcome": "retried"}
        feedback = (f"Harness check: your answer was rejected because {problem}. Answer again following "
                    f"AGENTS.md: cite only passages that search returned, exactly as shown, or reply "
                    f"exactly '{REFUSAL}'")
        result = await agent.ainvoke({"messages": messages + [HumanMessage(feedback)]})
        messages = result["messages"]
        answer = messages[-1].text.strip()
        problem = grounding_problem(answer, retrieved_passages(messages), action_request)
        if problem:
            guardrail["outcome"] = "replaced by a refusal"
            guardrail["rejected_answer"] = answer
            answer = f"{REFUSAL} (The answer I drafted could not be supported by the passages I found.)"

    return {
        "question": question,
        "answer": answer,
        "trace": tool_trace(messages),
        "retrieved": retrieved_passages(messages),
        "citations": citations(answer),
        "refusal": is_refusal(answer),
        "tokens": tokens_used(messages),
        "model_calls": sum(isinstance(m, AIMessage) for m in messages),
        "seconds": round(time.time() - started, 1),
        "guardrail": guardrail,
    }


def print_result(run: dict) -> None:
    for line in run["trace"]:
        first = line.splitlines()[0] if line else line
        print(f"  {first[:160]}", file=sys.stderr)
    if run["guardrail"]:
        print(f"  guardrail fired: {run['guardrail']['fired']} -> {run['guardrail']['outcome']}", file=sys.stderr)
    print(run["answer"])
    if run["citations"]:
        print("\nSources:")
        for source, section in run["citations"]:
            print(f"  [{source}, {section}]  {run['retrieved'].get((source, section), '')}")
    tokens = run["tokens"]
    print(f"\n({run['model_calls']} model calls, {tokens['input'] + tokens['output']:,} tokens, "
          f"{run['seconds']} s)", file=sys.stderr)


async def main(question: str) -> None:
    async with AsyncExitStack() as stack:
        agent = build_agent(await load_mcp_tools(stack))
        run = await ask(agent, question)
    print_result(run)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python agent.py "your question"')
    asyncio.run(main(" ".join(sys.argv[1:])))