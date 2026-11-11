"""
Assignment 3: build the index (`main.py index`).

Corpus: the EU drone rules (see SOURCES). The PDFs are committed in corpus/,
because EUR-Lex blocks scripted downloads and its licence allows reuse and
redistribution with attribution (Commission Decision 2011/833/EU).

    uv run python main.py index         # extract, chunk, embed what is missing, store
    uv run python index.py --dry-run    # extract and chunk only: no API calls

Two chunking designs are built so evaluate.py can compare them by hit rate:
  structure  one chunk per article or annex point (e.g. UAS.OPEN.030); long
             ones are split into overlapping windows; the heading path is
             part of the text that gets embedded
  fixed      250-word windows with a 50-word overlap, blind to structure

Every chunk carries its source, its location (heading path and PDF pages) and
the list of sections it overlaps, which evaluate.py uses to score hits.

Every chunk vector is appended to index/vectors.jsonl as soon as it arrives,
keyed by a hash of the exact text embedded. A crash or the daily quota loses
nothing, and a re-run embeds only chunks without a saved vector. Query vectors
go to index/query_vectors.jsonl, so asking the same question twice is free.

Per-file steps, imported by server.py's add-document tool:
  chunk_file(path, design)       extract, clean and chunk one file
  embed_and_store(chunks, design) embed what has no saved vector, store in ChromaDB
  add_file(path)                 both, with the already-indexed check
  search(query, k)               the k closest chunks, used by the search tool

Nothing in this module prints to stdout (the MCP server uses stdout for the
protocol): logs go to stderr.
"""

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent
CORPUS_DIR = ROOT / "corpus"
INDEX_DIR = ROOT / "index"
CHROMA_DIR = INDEX_DIR / "chroma"
VECTOR_CACHE = INDEX_DIR / "vectors.jsonl"         # chunk vectors
QUERY_CACHE = INDEX_DIR / "query_vectors.jsonl"    # question vectors
REGISTRY_FILE = INDEX_DIR / "documents.json"
STATS_FILE = INDEX_DIR / "corpus_stats.json"
SECTIONS_FILE = INDEX_DIR / "sections.txt"

EMBED_MODEL = "gemini-embedding-2"
EMBED_DIM = 768          # MRL truncation: a quarter of the storage, almost the same quality
EMBED_BATCH = 25         # texts per request; each text still counts towards the daily quota
DESIGNS = ("structure", "fixed")   # embedding order: the server's design first
CHOSEN_DESIGN = "structure"        # the design server.py searches; set from evaluate.py's hit rates
FIXED_WORDS, FIXED_OVERLAP = 250, 50
SECTION_MAX_WORDS = 350            # longer sections are split into FIXED_WORDS windows
SECTION_MIN_WORDS = 25             # shorter ones (a heading and its title) are carried into the next section
MIN_CORPUS_WORDS = 50_000
MAX_ADD_CHUNKS = 300               # add_file refuses bigger documents: they would eat the daily quota
SUPPORTED = {".pdf", ".txt", ".md"}

EUR_LEX_LICENCE = (
    "© European Union, EUR-Lex. Reuse authorised provided the source is acknowledged "
    "(Commission Decision 2011/833/EU)."
)
SOURCES = {
    "reg-2019-947": {
        "title": "Commission Implementing Regulation (EU) 2019/947 (operation of unmanned aircraft)",
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32019R0947",
        "licence": EUR_LEX_LICENCE,
    },
    "reg-2019-945": {
        "title": "Commission Delegated Regulation (EU) 2019/945 (unmanned aircraft systems and classes)",
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32019R0945",
        "licence": EUR_LEX_LICENCE,
    },
    "reg-2021-664": {
        "title": "Commission Implementing Regulation (EU) 2021/664 (U-space regulatory framework)",
        "url": "https://eur-lex.europa.eu/eli/reg_impl/2021/664/2026-02-22/eng/pdf",
        "licence": EUR_LEX_LICENCE,
    },
    # Consolidated texts omit the recitals ("Whereas: (1) ..."), which explain each rule.
    # They are taken from the original Official Journal publication; only the part
    # before Article 1 is kept, so none of its outdated articles enter the index.
    "reg-2019-947-recitals": {
        "title": "Recitals of Implementing Regulation (EU) 2019/947 (original Official Journal text)",
        "url": "https://eur-lex.europa.eu/eli/reg_impl/2019/947/oj/eng",
        "licence": EUR_LEX_LICENCE,
        "date": "11.6.2019 (OJ L 152)",
        "recitals_only": True,
    },
    "reg-2019-945-recitals": {
        "title": "Recitals of Delegated Regulation (EU) 2019/945 (original Official Journal text)",
        "url": "https://eur-lex.europa.eu/eli/reg_del/2019/945/oj/eng",
        "licence": EUR_LEX_LICENCE,
        "date": "11.6.2019 (OJ L 152)",
        "recitals_only": True,
    },
    "reg-2021-664-recitals": {
        "title": "Recitals of Implementing Regulation (EU) 2021/664 (original Official Journal text)",
        "url": "https://eur-lex.europa.eu/eli/reg_impl/2021/664/oj/eng",
        "licence": EUR_LEX_LICENCE,
        "date": "23.4.2021 (OJ L 139)",
        "recitals_only": True,
    },
    "rd-517-2024": {
        "title": "Real Decreto 517/2024 (Spain, civil use of unmanned aircraft systems)",
        "url": "https://boe.es/buscar/pdf/2024/BOE-A-2024-11377-consolidado.pdf",
        "licence": "Fuente de los datos: Agencia Estatal Boletín Oficial del Estado (BOE reuse conditions).",
        "date": "5.6.2024 (BOE núm. 136), consolidated text",
    },
}


def log(*parts) -> None:
    print(*parts, file=sys.stderr, flush=True)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------- extraction

def extract_pages(path: Path) -> list[str]:
    """The text of each page (one 'page' for a plain-text file).

    PDFs are read with PyMuPDF, not pypdf: EUR-Lex justifies its text with extra
    letter spacing, and pypdf turned that into broken words ("a ssemblies",
    "sh all": 80 in 2019/947 alone). PyMuPDF uses the character positions and
    extracts them whole.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        with pymupdf.open(path) as pdf:
            return [page.get_text() for page in pdf]
    if suffix in (".txt", ".md"):
        return [path.read_text(encoding="utf-8", errors="replace")]
    raise ValueError(f"Unsupported file type {suffix}; supported: {', '.join(sorted(SUPPORTED))}")


# EUR-Lex consolidated PDFs repeat a header on every page,
# e.g. "02019R0947 — EN — 01.05.2025 — 005.001 — 12".
EURLEX_HEADER = re.compile(
    r"^\d{5}[A-Z]\d{4}\s*[—–-]\s*[A-Z]{2}\s*[—–-]\s*\d{2}\.\d{2}\.\d{4}\s*[—–-]\s*\d{3}\.\d{3}\s*[—–-]\s*\d+\s*"
)
# Original Official Journal PDFs repeat a running header instead. Its parts come out
# of the PDF in a different order depending on the year:
#   2021: "23.4.2021 EN Official Journal of the European Union L 139/161"
#   2019: "11.6.2019 L 152/1 Official Journal of the European Union EN"
# Both forms need the date or the "L 152/1" page number, so the phrase
# "published in the Official Journal of the European Union" in the text is kept.
OJ_HEADER = re.compile(
    r"^(?:\d{1,2}\.\d{1,2}\.\d{4}\s+)?(?:L\s*\d+/\d+\s+)?EN\s+Official Journal of the European Union"
    r"(?:\s+L\s*\d+/\d+)?(?:\s+\d{1,2}\.\d{1,2}\.\d{4})?\s*"
    r"|\d{1,2}\.\d{1,2}\.\d{4}\s+L\s*\d+/\d+\s+Official Journal of the European Union(?:\s+EN\b)?\s*"
)
# Amendment markers in consolidated texts: ▼B, ▼M1, ►M2 … ◄, ►C1.
AMENDMENT_MARKERS = re.compile(r"[▼►](?:[MCA]\d{1,2}|B)?|◄")
# If the Official Journal header comes out split over several lines, its pieces are
# dropped one by one. (A lone date is kept: real tables contain them, 6 in 2019/947.)
# BOE consolidated PDFs repeat their last three lines on every page.
NOISE_LINES = [
    re.compile(r"^Official Journal of the European Union$"),
    re.compile(r"^L\s*\d+/\d+$"),
    re.compile(r"^EN$"),
    re.compile(r"^BOLETÍN OFICIAL DEL ESTADO$"),
    re.compile(r"^LEGISLACIÓN CONSOLIDADA$"),
    re.compile(r"^Página\s+\d+(\s+de\s+\d+)?$"),
]
LONE_DATE = re.compile(r"^\d{1,2}\.\d{1,2}\.\d{4}$")
ARTICLE_1 = re.compile(r"^Article 1$")
# The act's own title, e.g. "COMMISSION IMPLEMENTING REGULATION (EU) 2019/947".
ACT_TITLE = re.compile(r"^(?:COMMISSION (?:IMPLEMENTING |DELEGATED )?)?REGULATION \(EU\)")


def clean_lines(pages: list[str]) -> tuple[list[tuple[str, int]], dict]:
    """Clean text lines with their page number, and counts of what cleaning did."""
    lines, stats = [], {"headers_removed": 0, "noise_lines_removed": 0, "soft_hyphens_joined": 0}
    for page_no, page in enumerate(pages, start=1):
        # A soft hyphen (U+00AD) marks a word the typesetter broke at a line end:
        # "limi\xad" + "tations" is always one word, "limitations".
        page, n_soft = re.subn(r"\xad[ \t]*\n[ \t]*", "", page)
        stats["soft_hyphens_joined"] += n_soft
        page = page.replace("\xad", "")
        for line in page.splitlines():
            line = unicodedata.normalize("NFKC", line)        # ligatures: "ﬁ" -> "fi"
            header = EURLEX_HEADER.match(line)
            if header and "version_date" not in stats:   # "02019R0947 — EN — 01.05.2025 — ..."
                stats["version_date"] = re.findall(r"\d{2}\.\d{2}\.\d{4}", header.group(0))[0]
            line, n_headers = EURLEX_HEADER.subn("", line)
            line, n_oj = OJ_HEADER.subn("", line)
            stats["headers_removed"] += n_headers + n_oj
            line = AMENDMENT_MARKERS.sub("", line)
            line = re.sub(r"\s+", " ", line).strip()
            line = re.sub(r" ([.,;:])", r"\1", line)   # space left where a marker stood
            if not line:
                continue
            if any(pattern.match(line) for pattern in NOISE_LINES):
                stats["noise_lines_removed"] += 1
                # The header's date, split onto its own line just before, goes too.
                if lines and lines[-1][1] == page_no and LONE_DATE.match(lines[-1][0]):
                    lines.pop()
                    stats["noise_lines_removed"] += 1
                continue
            lines.append([line, page_no])

    # Skip front matter but keep the recitals. A consolidated EUR-Lex PDF opens with a
    # disclaimer and a list of amendments, then repeats the act's title before the recitals:
    # start at the LAST title line before Article 1. The BOE's table of contents ends at
    # "TEXTO CONSOLIDADO".
    start = next((i + 1 for i, (line, _) in enumerate(lines) if "TEXTO CONSOLIDADO" in line), None)
    if start is None:
        article_1 = next((i for i, (line, _) in enumerate(lines) if ARTICLE_1.match(line)), len(lines))
        titles = [i for i, (line, _) in enumerate(lines[:article_1]) if ACT_TITLE.match(line)]
        start = titles[-1] if titles else (article_1 if article_1 < len(lines) else 0)
    stats["front_matter_lines_skipped"] = start
    lines = lines[start:]

    # Words split across lines: "require-" + "ments". Join without the hyphen when the
    # joined word appears elsewhere in the document; otherwise keep it ("geo-awareness").
    vocabulary = {w.lower() for line, _ in lines for w in re.findall(r"[^\W\d_]+", line)}
    stats["hyphens_joined"] = stats["hyphens_kept"] = 0
    for i in range(len(lines) - 1):
        line, following = lines[i][0], lines[i + 1][0]
        tail = re.search(r"([^\W\d_]+)-$", line)
        if not tail or not following[:1].islower():
            continue
        head, _, rest = following.partition(" ")
        head_letters = re.match(r"[^\W\d_]+", head)
        candidate = tail.group(1) + (head_letters.group(0) if head_letters else "")
        if candidate.lower() in vocabulary:
            lines[i][0] = line[:-1] + head
            stats["hyphens_joined"] += 1
        else:
            lines[i][0] = line + head
            stats["hyphens_kept"] += 1
        lines[i + 1][0] = rest
    return [(line, page) for line, page in lines if line], stats


# ---------------------------------------------------------------- structure

# (level, label, pattern). A heading closes every open heading of the same or a deeper level.
# Whole-line patterns for title-case headings avoid false hits on cross-references
# that happen to start a line ("Article 5 of Regulation ...").
HEADINGS = [
    (0, "Annex", r"^ANNEX(?:\s+(?P<n>[IVXLC]+))?(?:\s|$)"),
    (0, "Annex", r"^Annex(?:\s+(?P<n>[IVXLC]+))?$"),
    (0, "Anexo", r"^ANEXO(?:\s+(?P<n>[IVXLC]+))?(?:\s|$|\.)"),
    (0, "Chapter", r"^CHAPTER\s+(?P<n>[IVXLC]+)(?:\s|$)"),
    (0, "Chapter", r"^Chapter\s+(?P<n>[IVXLC]+)$"),
    (0, "Capítulo", r"^CAPÍTULO\s+(?P<n>[IVXLC]+)(?:\s|$|\.)"),
    (1, "Part", r"^PART\s+(?P<n>[A-Z]|\d+)(?:\s|$)"),
    (1, "Part", r"^Part\s+(?P<n>[A-Z]|\d+)$"),
    (1, "Appendix", r"^APPENDIX\s+(?P<n>\d+|[IVXLC]+)(?:\s|$)"),
    (1, "Appendix", r"^Appendix\s+(?P<n>\d+|[IVXLC]+)$"),
    (2, "Section", r"^SECTION\s+(?P<n>\d+)(?:\s|$)"),
    (2, "Section", r"^Section\s+(?P<n>\d+)$"),
    (2, "Sección", r"^Sección\s+(?P<n>\d+\.[ªa])(?:\s|$)"),
    (3, "Recitals", r"^Whereas:?$"),       # same level as Article, so Article 1 closes it
    (3, "Article", r"^Article\s+(?P<n>\d+[a-z]?)$"),
    (3, "Artículo", r"^Artículo\s+(?P<n>\d+(?:\s+bis)?)\."),
    (3, "", r"^(?P<n>UAS\.[A-Z]+(?:-\d{2})?\.\d{3})\s+[A-Z]"),   # annex points: UAS.OPEN.030
    (0, "Disposición", r"^Disposición\s+(?P<n>(?:adicional|transitoria|derogatoria|final)\s+\w+)"),  # after the last chapter, not in it
]
HEADINGS = [(level, label, re.compile(pattern)) for level, label, pattern in HEADINGS]


def tag_sections(lines: list[tuple[str, int]]) -> list[tuple[str, int, str]]:
    """Split lines into words, each tagged with its page and section path ('Annex > Part A > UAS.OPEN.030')."""
    stack: list[tuple[int, str]] = []
    tokens = []
    for line, page in lines:
        for level, label, pattern in HEADINGS:
            match = pattern.match(line)
            if match:
                number = match.groupdict().get("n")
                name = f"{label} {number}".strip() if number else label
                # Inside an annex, chapters sit below appendices (2019/947: Annex >
                # Appendix 1 > Chapter I > UAS.STS-01.010), not at the top level.
                if label in ("Chapter", "Capítulo") and any(n.startswith(("Annex", "Anexo")) for _, n in stack):
                    level = 2
                stack = [entry for entry in stack if entry[0] < level] + [(level, name)]
                break
        section = " > ".join(name for _, name in stack) or "Preamble"
        tokens += [(word, page, section) for word in line.split()]
    return tokens


def load_document(path: Path) -> dict:
    """Extract, clean and tag one file. Its source id is the file name without extension."""
    source = path.stem.lower()
    info = SOURCES.get(source, {})
    pages = extract_pages(path)
    lines, cleaning = clean_lines(pages)
    tokens = tag_sections(lines)
    if info.get("recitals_only"):
        before_article_1 = []
        for token in tokens:
            if token[2] not in ("Preamble", "Recitals"):
                break
            before_article_1.append(token)
        cleaning["words_after_recitals_dropped"] = len(tokens) - len(before_article_1)
        tokens = before_article_1
    return {
        "source": source,
        "title": info.get("title", path.stem),
        "url": info.get("url", relative(path)),
        "licence": info.get("licence", "not recorded"),
        "date": info.get("date") or cleaning.get("version_date", "not recorded"),
        "file": relative(path),
        "pages": len(pages),
        "words": len(tokens),
        "tokens": tokens,
        "cleaning": cleaning,
    }


# ---------------------------------------------------------------- chunking

def windows(tokens: list, size: int, overlap: int):
    """Overlapping windows of `size` words that cover every word."""
    if len(tokens) <= size:
        yield tokens
        return
    start = 0
    while True:
        yield tokens[start:start + size]
        if start + size >= len(tokens):
            return
        start += size - overlap


def short_sections(sections: list[str]) -> str:
    labels = list(dict.fromkeys(s.split(" > ")[-1] for s in sections))
    if len(labels) > 3:
        labels = [labels[0], "…", labels[-1]]
    return ", ".join(labels)


def make_chunk(doc: dict, design: str, n: int, tokens: list, section: str = "", part: int = 1, parts: int = 1) -> dict:
    pages = [page for _, page, _ in tokens]
    first, last = min(pages), max(pages)
    page_text = f"p. {first}" if first == last else f"pp. {first}-{last}"
    sections = list(dict.fromkeys(s for _, _, s in tokens))
    if design == "structure":
        location = f"{section}, {page_text}" + (f" (part {part} of {parts})" if parts > 1 else "")
    else:
        location = f"{page_text} ({short_sections(sections)})"
    return {
        "id": f"{design}:{doc['source']}:{n:04d}",
        "source": doc["source"],
        "title": doc["title"],
        "design": design,
        "section": section,
        "sections": " | ".join(sections),
        "location": location,
        "page_start": first,
        "page_end": last,
        "words": len(tokens),
        "text": " ".join(word for word, _, _ in tokens),
    }


def structure_groups(tokens: list) -> list[tuple[str, int, int, list]]:
    """(section, part, parts, tokens) per chunk: one per section, long sections split."""
    sections: list[list] = []
    for token in tokens:
        if sections and sections[-1][0] == token[2]:
            sections[-1][1].append(token)
        else:
            sections.append([token[2], [token]])

    merged, carry = [], []
    for section, toks in sections:
        toks = carry + toks
        carry = []
        if len(toks) < SECTION_MIN_WORDS:
            carry = toks
            continue
        merged.append((section, toks))
    if carry:
        if merged:
            merged[-1] = (merged[-1][0], merged[-1][1] + carry)
        else:
            merged.append((carry[-1][2], carry))

    groups = []
    for section, toks in merged:
        parts = [toks] if len(toks) <= SECTION_MAX_WORDS else list(windows(toks, FIXED_WORDS, FIXED_OVERLAP))
        groups += [(section, i, len(parts), part) for i, part in enumerate(parts, start=1)]
    return groups


def chunk_document(doc: dict, design: str) -> list[dict]:
    if not doc["tokens"]:
        return []
    if design == "fixed":
        return [make_chunk(doc, design, n, toks)
                for n, toks in enumerate(windows(doc["tokens"], FIXED_WORDS, FIXED_OVERLAP))]
    if design == "structure":
        return [make_chunk(doc, design, n, toks, section, part, parts)
                for n, (section, part, parts, toks) in enumerate(structure_groups(doc["tokens"]))]
    raise ValueError(f"Unknown design {design!r}; expected one of {DESIGNS}")


def chunk_file(path: str | Path, design: str = CHOSEN_DESIGN) -> tuple[dict, list[dict]]:
    """Extract, clean and chunk one file: (document info, chunks)."""
    doc = load_document(Path(path))
    return doc, chunk_document(doc, design)


def embedding_text(chunk: dict) -> str:
    """Gemini Embedding 2's document format. The structure design also embeds the heading path."""
    heading = chunk["title"] if chunk["design"] == "fixed" else f"{chunk['title']} — {chunk['section']}"
    return f"title: {heading} | text: {chunk['text']}"


# ---------------------------------------------------------------- embedding

class QuotaExhausted(RuntimeError):
    pass


_client = None
_caches: dict[Path, dict[str, list[float]]] = {}


def gemini():
    global _client
    if _client is None:
        from google import genai

        from api_key import load_api_key
        _client = genai.Client(api_key=load_api_key())
    return _client


def cache_key(text: str) -> str:
    return hashlib.sha256(f"{EMBED_MODEL}|{EMBED_DIM}|{text}".encode("utf-8")).hexdigest()


def load_cache(path: Path = VECTOR_CACHE) -> dict[str, list[float]]:
    if path not in _caches:
        cache = _caches[path] = {}
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                    cache[record["key"]] = record["v"]
                except (json.JSONDecodeError, KeyError):
                    continue  # a line cut short by a crash; that text is simply embedded again
    return _caches[path]


def retry_delay(error: Exception) -> float | None:
    match = re.search(r"retry(?:Delay)?\W+(?:in\s+)?(\d+(?:\.\d+)?)s", str(error), re.IGNORECASE)
    return float(match.group(1)) + 1 if match else None


def call_embed(texts: list[str]) -> list[list[float]]:
    """One request. Each text is wrapped in its own Content: with gemini-embedding-2,
    a plain list of strings would come back as ONE aggregated vector."""
    from google.genai import errors, types

    contents = [types.Content(parts=[types.Part.from_text(text=t)]) for t in texts]
    config = types.EmbedContentConfig(output_dimensionality=EMBED_DIM)
    for attempt in range(7):
        try:
            result = gemini().models.embed_content(model=EMBED_MODEL, contents=contents, config=config)
            vectors = [list(e.values) for e in result.embeddings]
            if len(vectors) != len(texts):
                raise RuntimeError(f"Asked for {len(texts)} embeddings, got {len(vectors)}.")
            return vectors
        except errors.APIError as error:
            if error.code == 429 and "PerDay" in str(error):
                raise QuotaExhausted(
                    "Daily embedding quota used up. Everything embedded so far is saved in "
                    "index/vectors.jsonl; run the same command again after it resets (09:00 in Spain)."
                ) from error
            if error.code in (429, 500, 502, 503, 504) and attempt < 6:
                wait = retry_delay(error) or min(60, 5 * 2 ** attempt)
                log(f"  API error {error.code}; retrying in {wait:.0f}s")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("unreachable")


def embed_texts(texts: list[str], cache_path: Path = VECTOR_CACHE) -> list[list[float]]:
    """A vector per text, from the cache when possible. New vectors are saved batch by batch."""
    cache = load_cache(cache_path)
    keys = [cache_key(t) for t in texts]
    missing = list(dict.fromkeys(t for t, k in zip(texts, keys) if k not in cache))
    if missing:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if len(missing) > 1:
            log(f"Embedding {len(missing)} new text(s) with {EMBED_MODEL} ...")
        with cache_path.open("a", encoding="utf-8") as f:
            for i in range(0, len(missing), EMBED_BATCH):
                batch = missing[i:i + EMBED_BATCH]
                for text, vector in zip(batch, call_embed(batch)):
                    key, vector = cache_key(text), [round(x, 6) for x in vector]
                    cache[key] = vector
                    f.write(json.dumps({"key": key, "v": vector}) + "\n")
                f.flush()
                if len(missing) > 1:
                    log(f"  {min(i + EMBED_BATCH, len(missing))}/{len(missing)} saved")
    return [cache[k] for k in keys]


def embed_query(query: str) -> list[float]:
    """Gemini Embedding 2's query format for question answering (task_type is not supported)."""
    return embed_texts([f"task: question answering | query: {query}"], QUERY_CACHE)[0]


def count_missing(chunks: list[dict]) -> int:
    cache = load_cache()
    return len({cache_key(embedding_text(c)) for c in chunks} - cache.keys())


# ---------------------------------------------------------------- storage

_chroma = None


def collection(design: str):
    global _chroma
    if _chroma is None:
        import chromadb
        from chromadb.config import Settings
        _chroma = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
    return _chroma.get_or_create_collection(
        name=design, embedding_function=None, configuration={"hnsw": {"space": "cosine"}}
    )


METADATA_FIELDS = ("source", "title", "design", "section", "sections", "location", "page_start", "page_end", "words")


def embed_and_store(chunks: list[dict], design: str, prune_sources: set[str] | None = None) -> None:
    """Embed what has no saved vector, and write new or changed chunks to ChromaDB.
    With prune_sources, delete those sources' chunks that no longer exist."""
    col = collection(design)
    stored = col.get(ids=[c["id"] for c in chunks], include=["documents", "metadatas"])
    stored = {i: (d, m) for i, d, m in zip(stored["ids"], stored["documents"], stored["metadatas"])}
    todo = [c for c in chunks
            if stored.get(c["id"]) != (c["text"], {k: c[k] for k in METADATA_FIELDS})]
    vectors = embed_texts([embedding_text(c) for c in todo])
    for i in range(0, len(todo), 200):
        batch = todo[i:i + 200]
        col.upsert(
            ids=[c["id"] for c in batch],
            embeddings=vectors[i:i + 200],
            documents=[c["text"] for c in batch],
            metadatas=[{k: c[k] for k in METADATA_FIELDS} for c in batch],
        )
    if prune_sources:
        keep = {c["id"] for c in chunks}
        existing = col.get(where={"source": {"$in": sorted(prune_sources)}}, include=[])["ids"]
        stale = [i for i in existing if i not in keep]
        if stale:
            col.delete(ids=stale)


def load_registry() -> dict:
    return json.loads(REGISTRY_FILE.read_text(encoding="utf-8")) if REGISTRY_FILE.is_file() else {}


def save_registry(registry: dict) -> None:
    REGISTRY_FILE.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")


def register(doc: dict, path: Path, origin: str, chunks_by_design: dict[str, int]) -> None:
    registry = load_registry()
    previous = registry.get(doc["source"], {}).get("chunks", {})
    registry[doc["source"]] = {
        "title": doc["title"], "url": doc["url"], "date": doc["date"], "licence": doc["licence"], "file": doc["file"],
        "sha256": file_sha(path), "pages": doc["pages"], "words": doc["words"], "origin": origin,
        "chunks": {**previous, **chunks_by_design}, "indexed_at": now(),
    }
    save_registry(registry)


# ---------------------------------------------------------------- search and add (used by server.py)

def search(query: str, k: int = 5, design: str | None = None, sources: list[str] | None = None) -> list[dict]:
    """The k chunks closest to the query, best first, with their metadata and a cosine score.
    `sources` limits the search to those documents (evaluate.py scores the corpus only)."""
    col = collection(design or CHOSEN_DESIGN)
    total = col.count()
    if total == 0:
        raise RuntimeError("The index is empty. Run `uv run python main.py index` first.")
    result = col.query(query_embeddings=[embed_query(query)], n_results=min(k, total),
                       where={"source": {"$in": sources}} if sources else None,
                       include=["documents", "metadatas", "distances"])
    return [
        {"id": chunk_id, "score": round(1 - distance, 4), "text": text, **meta}
        for chunk_id, text, meta, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]


def add_file(path: str | Path, design: str | None = None) -> dict:
    """Run one file through the pipeline and store it in the searched collection.

    Same bytes already indexed (under any name): nothing is done.
    Same file name with new content: its chunks are replaced (a new version of the rules).
    """
    design = design or CHOSEN_DESIGN
    path = Path(path).expanduser()
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if not path.is_file():
        return {"status": "error", "message": f"No file at {path}."}
    if path.suffix.lower() not in SUPPORTED:
        return {"status": "error", "message": f"Unsupported file type; supported: {', '.join(sorted(SUPPORTED))}."}

    sha = file_sha(path)
    registry = load_registry()
    for source, info in registry.items():
        if info.get("sha256") == sha and design in info.get("chunks", {}):
            return {"status": "already_indexed", "source": source, "title": info["title"],
                    "chunks": info["chunks"][design], "message": "This exact file is already in the index; nothing was changed."}

    doc, chunks = chunk_file(path, design)
    if not chunks:
        return {"status": "error", "message": "No text could be extracted from this file (is it a scanned PDF?)."}
    if len(chunks) > MAX_ADD_CHUNKS:
        return {"status": "error", "message": f"{len(chunks)} chunks is more than the {MAX_ADD_CHUNKS} allowed "
                                              "per added document (the free tier embeds about 1,000 texts a day)."}
    replaced = doc["source"] in registry
    embed_and_store(chunks, design, prune_sources={doc["source"]})
    register(doc, path, origin="corpus" if path.parent == CORPUS_DIR else "added", chunks_by_design={design: len(chunks)})
    return {"status": "replaced" if replaced else "added", "source": doc["source"], "title": doc["title"],
            "words": doc["words"], "chunks": len(chunks), "design": design}


# ---------------------------------------------------------------- build (main.py index)

def corpus_files() -> list[Path]:
    files = sorted(p for p in CORPUS_DIR.glob("*") if p.suffix.lower() in SUPPORTED) if CORPUS_DIR.is_dir() else []
    if not files:
        raise SystemExit(
            "corpus/ is empty. Put the regulation PDFs there (see SOURCES in index.py):\n"
            "  reg-2019-947.pdf, reg-2019-945.pdf, reg-2021-664.pdf (current consolidated versions)\n"
            "  reg-2019-947-recitals.pdf, reg-2019-945-recitals.pdf, reg-2021-664-recitals.pdf (original OJ texts)"
        )
    return files


def write_outputs(docs: list[dict], chunks: dict[str, list[dict]]) -> dict:
    INDEX_DIR.mkdir(exist_ok=True)
    for design, design_chunks in chunks.items():
        with (INDEX_DIR / f"chunks_{design}.jsonl").open("w", encoding="utf-8") as f:
            for chunk in design_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    with SECTIONS_FILE.open("w", encoding="utf-8") as f:   # to check heading detection by eye
        for doc in docs:
            f.write(f"## {doc['source']}\n")
            counts: dict[str, int] = {}
            for _, _, section in doc["tokens"]:
                counts[section] = counts.get(section, 0) + 1
            f.writelines(f"{n:>6}  {section}\n" for section, n in counts.items())

    stats = {
        "documents": [
            {key: doc[key] for key in ("source", "title", "url", "date", "licence", "file", "pages", "words", "cleaning")}
            | {"chunks": {d: sum(c["source"] == doc["source"] for c in chunks[d]) for d in chunks}}
            for doc in docs
        ],
        "total_words": sum(doc["words"] for doc in docs),
        "designs": {
            d: {"chunks": len(cs), "mean_words": round(sum(c["words"] for c in cs) / max(len(cs), 1), 1),
                "max_words": max((c["words"] for c in cs), default=0)}
            for d, cs in chunks.items()
        },
        "embedding": {"model": EMBED_MODEL, "dimensions": EMBED_DIM},
        "built_at": now(),
    }
    STATS_FILE.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats


def print_summary(stats: dict) -> None:
    log(f"\n{'document':<23}{'pages':>7}{'words':>9}" + "".join(f"{d:>11}" for d in DESIGNS))
    for doc in stats["documents"]:
        log(f"{doc['source']:<23}{doc['pages']:>7}{doc['words']:>9,}" + "".join(f"{doc['chunks'][d]:>11}" for d in DESIGNS))
    log(f"{'total':<23}{'':>7}{stats['total_words']:>9,}" + "".join(f"{stats['designs'][d]['chunks']:>11}" for d in DESIGNS))
    for d in DESIGNS:
        s = stats["designs"][d]
        log(f"{d}: mean {s['mean_words']} words per chunk, max {s['max_words']}")
    for doc in stats["documents"]:
        log(f"{doc['source']}: {doc['cleaning']}")


def build(dry_run: bool = False) -> None:
    files = corpus_files()
    log(f"Reading {len(files)} file(s) from corpus/ ...")
    docs = [load_document(path) for path in files]
    chunks = {design: [c for doc in docs for c in chunk_document(doc, design)] for design in DESIGNS}
    stats = write_outputs(docs, chunks)
    print_summary(stats)

    if stats["total_words"] < MIN_CORPUS_WORDS:
        raise SystemExit(f"\nThe corpus has {stats['total_words']:,} words; the brief requires {MIN_CORPUS_WORDS:,}.")

    pending = {design: count_missing(chunks[design]) for design in DESIGNS}
    log(f"\nTexts still to embed: {pending} (free tier: about 1,000 a day).")
    if dry_run:
        log("Dry run: nothing embedded. Check index/sections.txt and index/chunks_*.jsonl.")
        return

    sources = {doc["source"] for doc in docs}
    for design in DESIGNS:
        log(f"\n[{design}]")
        embed_and_store(chunks[design], design, prune_sources=sources)
    for doc, path in zip(docs, files):
        register(doc, path, origin="corpus",
                 chunks_by_design={d: sum(c["source"] == doc["source"] for c in chunks[d]) for d in DESIGNS})
    size_mb = sum(p.stat().st_size for p in INDEX_DIR.rglob("*") if p.is_file()) / 1e6
    log(f"\nIndex built: {', '.join(f'{d} {collection(d).count()} chunks' for d in DESIGNS)}; index/ is {size_mb:.1f} MB.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the index in index/.")
    parser.add_argument("--dry-run", action="store_true", help="extract and chunk only; no API calls")
    args = parser.parse_args()
    try:
        build(dry_run=args.dry_run)
    except QuotaExhausted as error:
        log(f"\n{error}")
        sys.exit(2)