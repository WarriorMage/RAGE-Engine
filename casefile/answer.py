"""Answer a question from the case files, with verified citations.

Pipeline for one question:
  1. Retrieve    : hybrid search, top 10 unique sources (max 2 chunks each)  - 1 Voyage call
  2. Investigate : the LLM proposes up to 3 keyword follow-up searches for missing evidence
                   (codes, plates, names); they run as text_search          - no Voyage calls
  3. Answer      : the LLM answers ONLY from the evidence, citing [E1], [E2], ...
  4. Verify      : unknown evidence ids are dropped, each quote is checked against the chunk
                   text and located in the source body (for highlighting), [E3] -> [SOURCE-ID]
"""

import re
from dataclasses import dataclass, field

from pymongo.database import Database

from casefile.llm import chat_json, model_name
from casefile.search import Filters, hybrid_search, text_search

MAX_SOURCES = 10            # measured in Step 8: recall@10 = 87-90% vs 80-83% at 5
MAX_CHUNKS_PER_SOURCE = 2
MAX_FOLLOW_UPS = 3
FOLLOW_UP_RESULTS = 4       # chunks fetched per follow-up search
MAX_EVIDENCE = 20           # total chunks given to the answer step
SEARCH_DEPTH = 40           # chunks fetched by the first search before picking sources


# ---------------------------------------------------------------- data classes

@dataclass
class Citation:
    label: str               # "E3"
    chunk_id: str            # "ACCESS-001#000"
    source_id: str
    title: str
    doc_type: str
    quote: str
    quote_verified: bool     # True if the quote really appears in the chunk text
    quote_start: int | None  # character position of the quote in the source body
    quote_end: int | None


@dataclass
class Answer:
    question: str
    answer: str                          # answer text with [SOURCE-ID] citations
    citations: list[Citation]
    evidence: list[dict]                 # every chunk shown to the model, with its "label"
    follow_up_searches: list[str] = field(default_factory=list)
    model: str = ""

    @property
    def cited_sources(self) -> list[str]:
        seen: list[str] = []
        for c in self.citations:
            if c.source_id not in seen:
                seen.append(c.source_id)
        return seen


# ---------------------------------------------------------------- retrieval

def select_evidence(results: list[dict], max_sources: int = MAX_SOURCES,
                    per_source: int = MAX_CHUNKS_PER_SOURCE) -> list[dict]:
    """Keep results in rank order, at most `per_source` chunks from each of `max_sources` sources."""
    picked: list[dict] = []
    counts: dict[str, int] = {}
    for r in results:
        sid = r["source_id"]
        if sid not in counts and len(counts) >= max_sources:
            continue
        if counts.get(sid, 0) >= per_source:
            continue
        counts[sid] = counts.get(sid, 0) + 1
        picked.append(r)
    return picked


def label_evidence(chunks: list[dict], start: int = 1) -> list[dict]:
    """Give each chunk a short label (E1, E2, ...) used in the prompt and in citations."""
    return [{**c, "label": f"E{i}"} for i, c in enumerate(chunks, start)]


def format_evidence(evidence: list[dict]) -> str:
    blocks = []
    for e in evidence:
        date = e["date"].strftime("%Y-%m-%d") if hasattr(e.get("date"), "strftime") else e.get("date", "")
        blocks.append(
            f"[{e['label']}] {e['source_id']} | {e['title']} | {e['doc_type']} | dated {date} | by {e.get('author', '')}\n"
            f"{e['text']}"
        )
    return "\n\n".join(blocks)


# ---------------------------------------------------------------- step 2: investigate

FOLLOW_UP_SYSTEM = """You help an investigator search a collection of case files.
You will see a question and the evidence found so far.
Suggest up to 3 short KEYWORD searches that would find evidence still MISSING to answer the
question fully. Good searches are exact identifiers that appear in the evidence: badge numbers
(e.g. HB-9042), car plates, phone numbers, receipt numbers, door or camera ids, full names,
or a specific time. Each search should be 1-4 words.
Return an empty list if the evidence already covers the question.
Never invent identifiers that do not appear in the question or the evidence."""

FOLLOW_UP_SCHEMA = {
    "type": "object",
    "properties": {"searches": {"type": "array", "items": {"type": "string"}}},
    "required": ["searches"],
    "additionalProperties": False,
}


def plan_follow_ups(question: str, evidence: list[dict]) -> list[str]:
    result = chat_json(
        FOLLOW_UP_SYSTEM,
        f"QUESTION: {question}\n\nEVIDENCE SO FAR:\n{format_evidence(evidence)}",
        FOLLOW_UP_SCHEMA,
        "follow_up_searches",
    )
    searches = []
    for s in result.get("searches", []):
        s = " ".join(s.split())
        if s and s.lower() not in (x.lower() for x in searches):
            searches.append(s)
    return searches[:MAX_FOLLOW_UPS]


def run_follow_ups(db: Database, searches: list[str], evidence: list[dict],
                   filters: Filters | None) -> list[dict]:
    """Run keyword follow-up searches and append chunks not already in the evidence."""
    have = {e["_id"] for e in evidence}
    added: list[dict] = []
    for s in searches:
        for r in text_search(db, s, FOLLOW_UP_RESULTS, filters):
            if len(evidence) + len(added) >= MAX_EVIDENCE:
                return added
            if r["_id"] not in have:
                have.add(r["_id"])
                added.append(r)
    return added


# ---------------------------------------------------------------- step 3: answer

ANSWER_SYSTEM = """You are Case File, an assistant for investigators.
Answer the question using ONLY the evidence excerpts provided. Rules:
- Every factual sentence must cite the evidence it comes from, like [E3]. When a statement
  combines several pieces of evidence, cite all of them, like [E2][E7].
- Use no outside knowledge and do not guess. Witnesses can be wrong or lie: say who claims
  what, and point out when evidence contradicts a claim.
- Give exact times, codes and names as written in the evidence.
- Be concise: 3-8 sentences, or a short bullet list.
- If the evidence does not answer the question, say clearly that the case files do not
  contain the answer, and mention only what IS known, with citations.
For each citation, also give a short quote (a few words up to one sentence) copied EXACTLY,
character for character, from that evidence excerpt."""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"evidence_id": {"type": "string"}, "quote": {"type": "string"}},
                "required": ["evidence_id", "quote"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["answer", "citations"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------- step 4: verify

def locate_quote(text: str, quote: str) -> tuple[int, int] | None:
    """Find `quote` in `text`, ignoring differences in whitespace and letter case.
    Returns (start, end) character offsets in `text`, or None if not found."""
    words = quote.split()
    if not words:
        return None
    pattern = r"\s+".join(re.escape(w) for w in words)
    m = re.search(pattern, text, flags=re.IGNORECASE)
    return (m.start(), m.end()) if m else None


LABEL_PATTERN = re.compile(r"\[(E\d+)\]")


def verify_citations(raw_citations: list[dict], evidence: list[dict]) -> list[Citation]:
    by_label = {e["label"]: e for e in evidence}
    citations: list[Citation] = []
    seen: set[tuple[str, str]] = set()
    for c in raw_citations:
        label = c.get("evidence_id", "").strip().strip("[]")
        chunk = by_label.get(label)
        if chunk is None:                 # the model cited something it was never shown
            continue
        quote = " ".join(c.get("quote", "").split())
        if (label, quote) in seen:
            continue
        seen.add((label, quote))
        span = locate_quote(chunk["text"], quote)
        citations.append(Citation(
            label=label,
            chunk_id=chunk["_id"],
            source_id=chunk["source_id"],
            title=chunk["title"],
            doc_type=chunk["doc_type"],
            quote=quote,
            quote_verified=span is not None,
            quote_start=chunk["start_char"] + span[0] if span else None,
            quote_end=chunk["start_char"] + span[1] if span else None,
        ))
    return citations


def replace_labels(answer: str, evidence: list[dict]) -> str:
    """Turn [E3] into [ACCESS-001]; drop labels that do not exist; merge repeats like [A][A]."""
    by_label = {e["label"]: e["source_id"] for e in evidence}
    text = LABEL_PATTERN.sub(lambda m: f"[{by_label[m.group(1)]}]" if m.group(1) in by_label else "", answer)
    text = re.sub(r"(\[[A-Z]+-\d+\])(\1)+", r"\1", text)   # [ACCESS-001][ACCESS-001] -> [ACCESS-001]
    return re.sub(r"[ \t]+([.,;:])", r"\1", text).strip()


# ---------------------------------------------------------------- main entry point

def answer_question(db: Database, question: str, filters: Filters | None = None) -> Answer:
    # 1. Retrieve (the only Voyage call)
    first = hybrid_search(db, question, k=SEARCH_DEPTH, filters=filters, depth=SEARCH_DEPTH)
    evidence = label_evidence(select_evidence(first))

    # 2. Investigate with keyword follow-ups
    follow_ups: list[str] = []
    if evidence:
        follow_ups = plan_follow_ups(question, evidence)
        extra = run_follow_ups(db, follow_ups, evidence, filters)
        evidence += label_evidence(extra, start=len(evidence) + 1)

    if not evidence:
        return Answer(question, "The case files contain no evidence matching this question.",
                      [], [], follow_ups, model_name())

    # 3. Answer from the evidence only
    result = chat_json(
        ANSWER_SYSTEM,
        f"QUESTION: {question}\n\nEVIDENCE:\n{format_evidence(evidence)}",
        ANSWER_SCHEMA,
        "cited_answer",
    )

    # 4. Verify citations and swap labels for source ids
    citations = verify_citations(result.get("citations", []), evidence)
    return Answer(
        question=question,
        answer=replace_labels(result.get("answer", ""), evidence),
        citations=citations,
        evidence=evidence,
        follow_up_searches=follow_ups,
        model=model_name(),
    )
