"""
Assignment 3: MCP server. Your editor and agent.py start it; by hand: `uv run server.py`.

- search(query): the 5 passages closest to the query, each with its source and
  location, ready to cite.
- add_document(path): runs one file through index.py's pipeline (extract,
  clean, chunk, embed what has no saved vector) and stores it.

The index is loaded from index/ at startup and the corpus is never re-embedded:
search embeds only the question. stdout carries the MCP protocol, so nothing
is printed there; logs go to stderr.
"""

import contextlib
import sys

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

import index

TOP_K = 5   # passages per search; chosen from the hit rates in eval/retrieval.json

mcp = MCPServer("corpus", log_level="WARNING")

# Load the index once at startup, so a missing index is reported before the first call.
_loaded = index.collection(index.CHOSEN_DESIGN).count()
index.log(f"corpus server: {_loaded} passages in the '{index.CHOSEN_DESIGN}' index")


def citation(hit: dict) -> str:
    """The citation the agent must copy: [source, section]."""
    return f"[{hit['source']}, {hit['section']}]"


def failure(error: Exception) -> str:
    """A reason the model can act on (quota, key, network), not just "Error executing tool"."""
    if isinstance(error, index.QuotaExhausted):
        return str(error)
    return f"{type(error).__name__}: {error}"


def format_hits(query: str, hits: list[dict]) -> str:
    lines = [f"{len(hits)} passages for {query!r}, best match first. "
             "Cite each fact with the citation in brackets exactly as written."]
    for n, hit in enumerate(hits, start=1):
        lines += ["", f"{n}. {citation(hit)}",
                  f"   {hit['title']} | {hit['location']} | similarity {hit['score']:.2f}",
                  f"   {hit['text']}"]
    return "\n".join(lines)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def search(query: str) -> str:
    """Search the EU rules for flying drones (unmanned aircraft systems, UAS).

    The corpus holds three EU regulations in their current consolidated versions,
    plus their recitals (the reasons given for each rule):
    - Implementing Regulation (EU) 2019/947: how drones may be flown. Open,
      specific and certified categories; subcategories A1, A2 and A3; distances
      from people; maximum height; remote pilot age and competency; operator
      registration; geographical zones; standard scenarios STS-01 and STS-02.
    - Delegated Regulation (EU) 2019/945: drone classes C0 to C6 (mass, speed,
      low-speed mode, lights, remote identification, geo-awareness).
    - Implementing Regulation (EU) 2021/664: U-space airspace and its services.
    It also holds any document added with add_document (a Spanish one is in Spanish).

    Use it for every question about drone rules, before answering. Search again
    with other words, or once per part of a question, if the passages do not
    cover everything. If no passage states the answer, the corpus does not cover it.

    Returns the 5 passages closest to the query, best first. Each has a citation
    [source, section] to copy into the answer, the document title, the pages,
    a similarity score from 0 to 1, and the passage text.
    """
    try:
        with contextlib.redirect_stdout(sys.stderr):   # keep stray prints off the protocol
            hits = index.search(query, k=TOP_K)
    except Exception as error:
        raise ToolError(f"Search failed: {failure(error)}") from error
    return format_hits(query, hits)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True))
def add_document(path: str) -> str:
    """Add a document to the searchable corpus.

    Takes the path of a PDF, .txt or .md file on this computer (absolute, or
    relative to the project folder). The file goes through the same pipeline as
    the corpus: text extraction, cleaning, chunking by article or section, and
    embedding with Gemini. Its passages are then stored in the index, and search
    finds them from the next call on.

    If the identical file is already indexed, nothing changes. If a file with the
    same name but different content was indexed before, its old passages are
    replaced by the new ones. Files over 300 passages are refused, because each
    passage counts towards the daily embedding quota.

    Use it only when the user asks to add a document. Returns what happened:
    added, replaced, already indexed or an error, with the document's source id
    (the name to cite) and its number of passages.
    """
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = index.add_file(path)
    except Exception as error:   # vectors embedded so far are saved; a retry continues from there
        raise ToolError(f"Not added: {failure(error)}") from error
    status = result["status"]
    if status == "error":
        return f"Not added: {result['message']}"
    if status == "already_indexed":
        return (f"Already indexed: {result['source']} ({result['title']}), {result['chunks']} passages. "
                "Nothing was changed.")
    verb = "Replaced the earlier version of" if status == "replaced" else "Added"
    return (f"{verb} {result['source']} ({result['title']}): {result['words']:,} words in "
            f"{result['chunks']} passages. Search finds them now; cite them as [{result['source']}, <section>].")


if __name__ == "__main__":
    mcp.run()  # stdio