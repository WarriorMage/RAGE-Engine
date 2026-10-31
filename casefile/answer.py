"""Answer a question from the case files, with verified citations.

Pipeline for one question:
  1. Retrieve    : hybrid search, top 10 unique sources (max 2 chunks each)  - 1 Voyage call
  2. Investigate : the LLM proposes up to 3 keyword follow-up searches for missing evidence
                   (codes, plates, names); they run as text_search          - no Voyage calls
  3. Answer      : the LLM answers ONLY from the evidence, citing [E1], [E2], ...
  4. Verify      : unknown evidence ids are dropped, each quote is checked against the chunk
                   text and located in the source body (for highlighting), [E3] -> [SOURCE-ID]
  5. Guard       : (Step 10) every answer gets a status - answered / partial / conflicting /
                   insufficient - plus explicit contradictions and what is missing. Code-level
                   checks downgrade answers whose citations do not hold up.
"""

import re
from dataclasses import dataclass, field

from pymongo.database import Database

from casefile.llm import chat_json, last_model_used, model_name
from casefile.search import Filters, hybrid_search, text_search

MAX_SOURCES = 10            # measured in Step 8: recall@10 = 87-90% vs 80-83% at 5
MAX_CHUNKS_PER_SOURCE = 2
MAX_FOLLOW_UPS = 3
FOLLOW_UP_RESULTS = 6       # chunks fetched per follow-up search
MAX_EVIDENCE = 24           # total chunks given to the answer step
SEARCH_DEPTH = 40           # chunks fetched by the first search before picking sources

STATUSES = ["answered", "partial", "conflicting", "insufficient"]
STATUS_LABELS = {
    "answered": "✅ Answered",
    "partial": "🟡 Partly answered",
    "conflicting": "⚠️ Conflicting evidence",
    "insufficient": "❌ Not in the case files",
}
# Police notes summarise other documents; the original records are better evidence.
SUMMARY_TYPES = {"police_note"}


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
class Conflict:
    topic: str
    claim: str                    # what one source says
    claim_sources: list[str]      # source ids
    contradiction: str            # what other evidence shows instead
    contradiction_sources: list[str]
    assessment: str               # which side the evidence supports better, or "unresolved"


@dataclass
class Answer:
    question: str
    answer: str                          # answer text with [SOURCE-ID] citations
    citations: list[Citation]
    evidence: list[dict]                 # every chunk shown to the model, with its "label"
    follow_up_searches: list[str] = field(default_factory=list)
    model: str = ""
    status: str = "answered"             # one of STATUSES
    conflicts: list[Conflict] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)    # what the case files do not say
    question_parts: list[dict] = field(default_factory=list)  # [{"part": ..., "covered": yes/partly/no}]
    warnings: list[str] = field(default_factory=list)   # problems found by the code checks

    @property
    def status_label(self) -> str:
        return STATUS_LABELS.get(self.status, self.status)

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
        kind = "SUMMARY of other documents" if e["doc_type"] in SUMMARY_TYPES else "original record"
        blocks.append(
            f"[{e['label']}] {e['source_id']} | {e['title']} | {e['doc_type']} ({kind}) | dated {date} | by {e.get('author', '')}\n"
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
Answer the question using ONLY the evidence excerpts provided.

CITATIONS
- Every factual sentence must cite the evidence it comes from, like [E3]. When a statement
  combines several pieces of evidence, cite all of them, like [E2][E7].
- Prefer ORIGINAL RECORDS (logs, CCTV, statements, interviews, emails, receipts, forensic
  reports). Cite a "SUMMARY" (police note) only if no original record shows the same fact.
- For each citation give a short quote (a few words up to one sentence) copied EXACTLY,
  character for character, from that evidence excerpt.

HONESTY
- Use no outside knowledge. Never guess or fill gaps.
- A reason, motive, intention, current location, or the content of a conversation counts as
  known ONLY if a document states it explicitly. Do not infer it from circumstances.
- A last known position is not a current position. A suspicious pattern is not proof.
- People can be mistaken or lie. Report claims as claims ("X says ..."), and check them
  against records such as CCTV, door logs and phone records.
- Give exact times, codes and names as written in the evidence.
- Be concise: 3-8 sentences, or a short bullet list.
- If the case files do not contain the answer, start with "The case files do not show ...",
  then give only closely related facts that ARE known, with citations.

FILL IN THE FIELDS IN THIS ORDER
1. "answer": the cited answer.
2. "question_parts": split the question into the things it asks (usually 1-3). For each,
   "covered" is "yes" if a document explicitly answers it, "partly" if only some of it is
   answered, "no" if the case files do not contain it.
3. "conflicts": list EVERY disagreement in the evidence about what the question asks, EVEN IF
   the records resolve it. Examples: a person's statement vs CCTV or door logs; two witnesses
   describing the same moment differently; someone changing their story between statement
   and interview. Give the claim and its evidence ids, what the other evidence shows and its
   evidence ids, and an assessment of which side the records support (or "unresolved").
   Use [] only if every relevant source agrees.
4. "missing": short phrases for information the case files do not contain. [] if none.
5. "citations": evidence id + exact quote for each citation used."""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "question_parts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "part": {"type": "string"},
                    "covered": {"type": "string", "enum": ["yes", "partly", "no"]},
                },
                "required": ["part", "covered"],
                "additionalProperties": False,
            },
        },
        "conflicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "claim": {"type": "string"},
                    "claim_evidence": {"type": "array", "items": {"type": "string"}},
                    "contradiction": {"type": "string"},
                    "contradiction_evidence": {"type": "array", "items": {"type": "string"}},
                    "assessment": {"type": "string"},
                },
                "required": ["topic", "claim", "claim_evidence", "contradiction",
                             "contradiction_evidence", "assessment"],
                "additionalProperties": False,
            },
        },
        "missing": {"type": "array", "items": {"type": "string"}},
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
    "required": ["answer", "question_parts", "conflicts", "missing", "citations"],
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


def labels_to_sources(labels: list[str], evidence: list[dict]) -> list[str]:
    """['E3', '[E5]', 'E99'] -> ['ACCESS-001', 'CCTV-003'] (unknown labels dropped, no repeats)."""
    by_label = {e["label"]: e["source_id"] for e in evidence}
    out: list[str] = []
    for label in labels:
        sid = by_label.get(label.strip().strip("[]"))
        if sid and sid not in out:
            out.append(sid)
    return out


def build_conflicts(raw: list[dict], evidence: list[dict]) -> list[Conflict]:
    conflicts = []
    for c in raw:
        claim_sources = labels_to_sources(c.get("claim_evidence", []), evidence)
        counter_sources = labels_to_sources(c.get("contradiction_evidence", []), evidence)
        if not (claim_sources or counter_sources):      # a conflict with no evidence at all is dropped
            continue
        conflicts.append(Conflict(
            topic=c.get("topic", ""),
            claim=replace_labels(c.get("claim", ""), evidence),
            claim_sources=claim_sources,
            contradiction=replace_labels(c.get("contradiction", ""), evidence),
            contradiction_sources=counter_sources,
            assessment=replace_labels(c.get("assessment", ""), evidence),
        ))
    return conflicts


def decide_status(parts: list[dict], citations: list[Citation],
                  conflicts: list[Conflict]) -> tuple[str, list[str]]:
    """The verdict is decided HERE, by fixed rules, not by the model.

    The model only reports facts about its answer (which parts are covered, which conflicts it
    saw); weaker or rushed models are unreliable at choosing a label themselves.
      every question part "no"         -> insufficient
      no valid citations               -> insufficient
      any conflict listed              -> conflicting
      some part "no" or "partly"       -> partial
      otherwise                        -> answered
    Quotes that could not be found in their source add a warning.
    """
    warnings: list[str] = []
    covered = [p.get("covered", "no") for p in parts] or ["no"]
    if all(c == "no" for c in covered):
        status = "insufficient"
    elif not citations:
        warnings.append("The answer had no valid citations, so it is not trusted.")
        status = "insufficient"
    elif conflicts:
        status = "conflicting"
    elif any(c != "yes" for c in covered):
        status = "partial"
    else:
        status = "answered"
    unverified = [c for c in citations if not c.quote_verified]
    if unverified:
        warnings.append(f"{len(unverified)} of {len(citations)} quotes could not be found word-for-word "
                        f"in their source ({', '.join(sorted({c.source_id for c in unverified}))}).")
    return status, warnings


def replace_labels(answer: str, evidence: list[dict]) -> str:
    """Turn [E3] into [ACCESS-001]; drop labels that do not exist; merge repeats like [A][A]."""
    by_label = {e["label"]: e["source_id"] for e in evidence}
    text = LABEL_PATTERN.sub(lambda m: f"[{by_label[m.group(1)]}]" if m.group(1) in by_label else "", answer)
    # [ACCESS-001][ACCESS-001] or [ACCESS-001], [ACCESS-001] -> [ACCESS-001]
    text = re.sub(r"(\[[A-Z]+-\d+\])(?:\s*,?\s*\1)+", r"\1", text)
    return re.sub(r"[ \t]+([.,;:])", r"\1", text).strip()


# ---------------------------------------------------------------- main entry point

def answer_question(db: Database, question: str, filters: Filters | None = None,
                    query_vector: list[float] | None = None) -> Answer:
    # 1. Retrieve (the only Voyage call, skipped if a query vector is given)
    first = hybrid_search(db, question, k=SEARCH_DEPTH, filters=filters, depth=SEARCH_DEPTH,
                          query_vector=query_vector)
    evidence = label_evidence(select_evidence(first))

    # 2. Investigate with keyword follow-ups
    follow_ups: list[str] = []
    if evidence:
        try:
            follow_ups = plan_follow_ups(question, evidence)
        except RuntimeError as e:   # optional step: answer anyway with the first evidence
            print(f"  (skipping follow-up searches: {e})")
        extra = run_follow_ups(db, follow_ups, evidence, filters)
        evidence += label_evidence(extra, start=len(evidence) + 1)

    if not evidence:
        return Answer(question, "The case files contain no evidence matching this question.",
                      [], [], follow_ups, model_name(), status="insufficient",
                      missing=["any document matching this question"])

    # 3. Answer from the evidence only
    result = chat_json(
        ANSWER_SYSTEM,
        f"QUESTION: {question}\n\nEVIDENCE:\n{format_evidence(evidence)}",
        ANSWER_SCHEMA,
        "cited_answer",
    )

    # 4. Verify citations and swap labels for source ids
    citations = verify_citations(result.get("citations", []), evidence)
    conflicts = build_conflicts(result.get("conflicts", []), evidence)

    # 5. Honesty guard: the code decides the verdict
    parts = result.get("question_parts", [])
    status, warnings = decide_status(parts, citations, conflicts)
    return Answer(
        question=question,
        answer=replace_labels(result.get("answer", ""), evidence),
        citations=citations,
        evidence=evidence,
        follow_up_searches=follow_ups,
        model=last_model_used(),
        status=status,
        question_parts=parts,
        conflicts=conflicts,
        missing=[m for m in result.get("missing", []) if m.strip()],
        warnings=warnings,
    )
