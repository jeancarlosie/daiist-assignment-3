"""
Assignment 3: evaluation (`main.py evaluate`).

  1. Retrieval   hit rate at k = 1, 3, 5 for both chunking designs, and the
                 words each design adds to the prompt         -> eval/retrieval.json
  2. Agent       agent.py on every question in eval/questions.csv
                                                               -> eval/answers.jsonl
  3. Code checks citations valid (each one names a passage search returned),
                 refusal or not, expected passage retrieved and cited
  4. Judge       a stronger Gemini model grades correctness against the
                 reference answer and support against the retrieved text
                                                               -> eval/verdicts.jsonl
  5. Agreement   judge vs your grades in eval/human_labels.csv
  6. Summary     outcomes, failure causes, cost per question
                 -> eval/results.csv, eval/summary.json, eval/grading_sheet.md

Answers and verdicts are saved one by one and reused on the next run, so a
crash or the daily quota loses nothing, and grading your 20+ answers does not
change them. Delete eval/answers.jsonl (and eval/verdicts.jsonl) to start over.

eval/grading_sheet.md shows each answer with its passages but not the judge's
verdict: grade from it before opening eval/verdicts.jsonl or eval/results.csv.
"""

import asyncio
import csv
import json
import re
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

import index
from agent import ask, build_agent, load_mcp_tools

ROOT = Path(__file__).resolve().parent
EVAL = ROOT / "eval"
QUESTIONS = EVAL / "questions.csv"
HUMAN_LABELS = EVAL / "human_labels.csv"
ANSWERS = EVAL / "answers.jsonl"
VERDICTS = EVAL / "verdicts.jsonl"
JUDGE_MODEL = "gemini-3.5-flash"  # a stronger model than the agent's
KS = (1, 3, 5)
PAUSE = 4  # seconds between questions, to stay under the free tier's requests per minute
MAX_WAIT = 300  # a longer wait means a daily quota: stop and resume after the reset instead


class DailyQuota(RuntimeError):
    pass


def log(*parts) -> None:
    print(*parts, flush=True)


def read_questions() -> list[dict]:
    with QUESTIONS.open(newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["question"].strip()]


def expected_passages(q: dict) -> list[tuple[str, str]]:
    """[(source, location)], one per passage the question needs; location may end in ', p. N'."""
    if q["kind"] == "unanswerable":
        return []
    return list(zip(q["expected_source"].split(";"), q["expected_location"].split(";")))


def pages_of(location: str) -> tuple[int, int] | None:
    found = re.search(r"pp?\. (\d+)(?:-(\d+))?", location)
    return (int(found[1]), int(found[2] or found[1])) if found else None


def matches(source: str, sections: list[str], location: str, expected: tuple[str, str]) -> bool:
    """Does a passage (its source, the sections it overlaps, its location) satisfy one expected passage?"""
    want_source, want = expected
    section, _, page = want.strip().partition(", p. ")
    if source != want_source.strip() or section not in sections:
        return False
    if not page:
        return True
    span = pages_of(location)
    return span is not None and span[0] <= int(page) <= span[1]


def found_all(passages: list[dict], q: dict) -> bool:
    return all(any(matches(p["source"], p["sections"], p["location"], e) for p in passages)
               for e in expected_passages(q))


# ---------------------------------------------------------------- 1. retrieval

def corpus_sources() -> list[str]:
    """The documents indexed by main.py index; documents added later with add_document are left out."""
    registry = index.load_registry()
    return sorted(source for source, info in registry.items() if info.get("origin") == "corpus")


def evaluate_retrieval(questions: list[dict]) -> dict:
    """Hit rate at k for each design: a hit when ALL expected passages are among the top k.
    Scored over the corpus only, so the numbers do not move when a document is added later."""
    answerable = [q for q in questions if q["kind"] != "unanswerable"]
    sources = corpus_sources()
    results = {}
    for design in index.DESIGNS:
        hits = {k: 0 for k in KS}
        words = {k: 0 for k in KS}
        per_question = {}
        for q in answerable:
            top = index.search(q["question"], k=max(KS), design=design, sources=sources)
            for p in top:
                p["sections"] = p["sections"].split(" | ")
            first_hit = next((k for k in KS if found_all(top[:k], q)), None)
            per_question[q["id"]] = first_hit
            for k in KS:
                hits[k] += first_hit is not None and first_hit <= k
                words[k] += sum(p["words"] for p in top[:k])
        n = len(answerable)
        results[design] = {
            "hit_rate": {f"@{k}": round(hits[k] / n, 3) for k in KS},
            "mean_words_added": {f"@{k}": round(words[k] / n) for k in KS},
            "first_hit_k": per_question,
            "questions": n,
        }
    # Choose by the numbers: hit@5, then hit@3, then hit@1, then fewer words added to the prompt.
    best = max(index.DESIGNS, key=lambda d: (results[d]["hit_rate"]["@5"], results[d]["hit_rate"]["@3"],
                                            results[d]["hit_rate"]["@1"], -results[d]["mean_words_added"]["@5"]))
    results["best_design"] = best
    results["design_in_use"] = index.CHOSEN_DESIGN
    (EVAL / "retrieval.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    log(f"\n{'design':<12}" + "".join(f"{'hit@' + str(k):>9}" for k in KS) + "".join(f"{'words@' + str(k):>10}" for k in KS))
    for design in index.DESIGNS:
        r = results[design]
        log(f"{design:<12}" + "".join(f"{r['hit_rate'][f'@{k}']:>9.0%}" for k in KS)
            + "".join(f"{r['mean_words_added'][f'@{k}']:>10}" for k in KS))
    log(f"Best by the numbers: {best}. The server uses: {index.CHOSEN_DESIGN} (CHOSEN_DESIGN in index.py).")
    if best != index.CHOSEN_DESIGN:
        raise SystemExit(f"\nThe numbers favour '{best}'. Set CHOSEN_DESIGN = \"{best}\" in index.py and run "
                         "again; stopping before the agent runs on the design the numbers did not choose.")
    return results


# ---------------------------------------------------------------- 2. agent on every question

def load_jsonl(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["id"]: row for row in rows}


def is_rate_limit(error: Exception) -> bool:
    return "429" in str(error) or "RESOURCE_EXHAUSTED" in str(error)


def wait_for(error: Exception) -> float:
    found = re.search(r"retry(?:Delay)?\W+(?:in\s+)?(\d+(?:\.\d+)?)s", str(error), re.IGNORECASE)
    return float(found[1]) + 2 if found else 60


async def with_retries(make_call, what: str):
    for attempt in range(6):
        try:
            return await make_call()
        except Exception as error:   # noqa: BLE001 - rate limits come as several error types
            if not is_rate_limit(error) or attempt == 5:
                raise
            delay = wait_for(error)
            if delay > MAX_WAIT:
                raise DailyQuota(
                    f"Daily quota reached on {what} (Google asks to wait {delay / 3600:.1f} h). Everything so far "
                    f"is saved in eval/; run `main.py evaluate` again after the reset (09:00 in Spain) to continue."
                ) from error
            log(f"    rate limit on {what}; waiting {delay:.0f}s")
            await asyncio.sleep(delay)


async def run_agent(questions: list[dict]) -> dict[str, dict]:
    answers = load_jsonl(ANSWERS)
    todo = [q for q in questions if answers.get(q["id"], {}).get("question") != q["question"]]
    log(f"\nAgent: {len(questions) - len(todo)} answers reused from {ANSWERS.name}, {len(todo)} to run.")
    if not todo:
        return answers
    async with AsyncExitStack() as stack:
        agent = build_agent(await load_mcp_tools(stack), approve=lambda call: False)  # never adds documents
        for q in todo:
            run = await with_retries(lambda: ask(agent, q["question"]), q["id"])
            run["retrieved"] = [{"source": s, "section": sec, "detail": d} for (s, sec), d in run["retrieved"].items()]
            run["citations"] = [list(c) for c in run["citations"]]
            answers[q["id"]] = {"id": q["id"], **run}
            with ANSWERS.open("a", encoding="utf-8") as f:
                f.write(json.dumps(answers[q["id"]], ensure_ascii=False) + "\n")
            log(f"  {q['id']} {'refusal' if run['refusal'] else 'answer'}  {run['seconds']:>5}s  "
                f"{run['tokens']['input'] + run['tokens']['output']:>6} tokens"
                + (f"  guardrail: {run['guardrail']['outcome']}" if run["guardrail"] else ""))
            await asyncio.sleep(PAUSE)
    return answers


# ---------------------------------------------------------------- 3. code checks

def code_checks(q: dict, run: dict) -> dict:
    retrieved = [{"source": p["source"], "sections": [p["section"]],
                  "location": p["detail"].split(" | ")[1] if " | " in p["detail"] else ""} for p in run["retrieved"]]
    cited = [{"source": s, "sections": [sec],
              "location": next((p["location"] for p in retrieved if (p["source"], p["sections"][0]) == (s, sec)), "")}
             for s, sec in run["citations"]]
    valid = all(any((r["source"], r["sections"][0]) == (s, sec) for r in retrieved) for s, sec in run["citations"])
    return {
        "searched": any(line.startswith("call search") for line in run["trace"]),
        "refusal": run["refusal"],
        "n_citations": len(run["citations"]),
        "citations_valid": valid,
        "cites_something": bool(run["citations"]) or run["refusal"],
        "retrieved_expected": found_all(retrieved, q) if q["kind"] != "unanswerable" else None,
        "cited_expected": found_all(cited, q) if q["kind"] != "unanswerable" else None,
    }


# ---------------------------------------------------------------- 4. judge

class Verdict(BaseModel):
    reasoning: str = Field(description="Two to four sentences checking the answer against each criterion.")
    correct: bool = Field(description="Criterion 1.")
    supported: bool = Field(description="Criterion 2.")


JUDGE_PROMPT = """You grade an assistant that answers questions about EU drone rules using only passages it retrieved with a search tool.

Criteria:
1. correct
   - Answerable question: the answer gives the key facts of the reference answer (numbers with units, conditions, exceptions) and contradicts none of them. Missing minor detail is fine; a wrong or missing key number or condition is not. Declining to answer is NOT correct.
   - Unanswerable question (the reference says the documents do not cover it): correct only if the answer says the documents do not cover it and does not present an answer to the question. Mentioning related rules that the passages state is fine.
2. supported
   - Every factual claim in the answer is stated in the retrieved passages below. Judge this only against the passages, not against your own knowledge or the reference answer. A refusal that makes no other claim is supported.

Question ({kind}): {question}
Reference answer: {reference}

Retrieved passages (everything the search tool returned):
{evidence}

Answer to grade:
{answer}"""


def evidence_of(run: dict) -> str:
    results = [line.removeprefix("result ") for line in run["trace"] if line.startswith("result ")]
    return "\n\n".join(results) or "(the assistant did not search)"


async def run_judge(questions: list[dict], answers: dict[str, dict]) -> dict[str, dict]:
    verdicts = load_jsonl(VERDICTS)
    stale = [q for q in questions
             if verdicts.get(q["id"], {}).get("answer") != answers[q["id"]]["answer"]]
    log(f"\nJudge ({JUDGE_MODEL}): {len(questions) - len(stale)} verdicts reused, {len(stale)} to grade.")
    if not stale:
        return verdicts
    judge = ChatGoogleGenerativeAI(model=JUDGE_MODEL, temperature=0).with_structured_output(Verdict, include_raw=True)
    for q in stale:
        run = answers[q["id"]]
        prompt = JUDGE_PROMPT.format(kind=q["kind"], question=q["question"], reference=q["reference_answer"],
                                     evidence=evidence_of(run), answer=run["answer"])
        started = time.time()
        out = await with_retries(lambda: judge.ainvoke(prompt), f"judge {q['id']}")
        usage = getattr(out["raw"], "usage_metadata", None) or {}
        verdicts[q["id"]] = {"id": q["id"], "answer": run["answer"], **out["parsed"].model_dump(),
                             "tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                             "seconds": round(time.time() - started, 1)}
        with VERDICTS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(verdicts[q["id"]], ensure_ascii=False) + "\n")
        log(f"  {q['id']} graded")
        await asyncio.sleep(PAUSE)
    return verdicts


# ---------------------------------------------------------------- 5. outcomes, failures, agreement

def outcome(q: dict, checks: dict, verdict: dict) -> str:
    if q["kind"] == "unanswerable":
        if checks["refusal"]:
            return "correct refusal"
        return "answered (should refuse)"
    if checks["refusal"]:
        return "wrong refusal"
    if not verdict["supported"] or not checks["citations_valid"]:
        return "unsupported"
    return "correct" if verdict["correct"] else "incorrect"


def suggested_cause(q: dict, checks: dict, result: str) -> str:
    """First guess at the failure cause, from the code checks. Review each one by hand:
    'bad test question' can only be decided by reading the question and the passages."""
    if result in ("correct", "correct refusal"):
        return ""
    if result == "answered (should refuse)":
        return "unsupported claim"
    if not checks["retrieved_expected"]:
        return "retrieval miss"
    if result == "wrong refusal":
        return "wrong refusal"
    if result == "unsupported":
        return "unsupported claim"
    return "wrong answer from right passage"


def agreement(verdicts: dict[str, dict]) -> dict | None:
    if not HUMAN_LABELS.is_file():
        return None
    with HUMAN_LABELS.open(newline="", encoding="utf-8") as f:
        labels = [r for r in csv.DictReader(f) if (r.get("id") or "").strip() in verdicts]
    if not labels:
        return None

    def as_bool(value: str) -> bool:
        return value.strip() in ("1", "true", "True", "yes")

    rows = []
    for label in labels:
        qid = label["id"].strip()
        human = (as_bool(label["correct"]), as_bool(label["supported"]))
        judge = (bool(verdicts[qid]["correct"]), bool(verdicts[qid]["supported"]))
        rows.append({"id": qid, "human_correct": human[0], "judge_correct": judge[0],
                     "human_supported": human[1], "judge_supported": judge[1]})
    n = len(rows)
    same = lambda key: sum(r[f"human_{key}"] == r[f"judge_{key}"] for r in rows)  # noqa: E731
    return {
        "graded": n,
        "agree_correct": round(same("correct") / n, 3),
        "agree_supported": round(same("supported") / n, 3),
        "agree_both": round(sum(r["human_correct"] == r["judge_correct"] and r["human_supported"] == r["judge_supported"]
                                for r in rows) / n, 3),
        "disagreements": [r for r in rows if r["human_correct"] != r["judge_correct"]
                          or r["human_supported"] != r["judge_supported"]],
    }


# ---------------------------------------------------------------- 6. outputs

def write_grading_sheet(questions: list[dict], answers: dict[str, dict]) -> None:
    """Everything needed to grade an answer by hand, and nothing about the judge's verdict."""
    out = ["# Grading sheet", "",
           "Grade each answer in eval/human_labels.csv (id, correct, supported, notes) before opening "
           "eval/verdicts.jsonl or eval/results.csv.", "",
           "- correct = 1 if it matches the reference answer, or properly says the documents do not cover it",
           "- supported = 1 if every claim is in the passages the assistant retrieved", ""]
    for q in questions:
        run = answers[q["id"]]
        out += [f"## {q['id']} ({q['kind']})", "", f"**Question:** {q['question']}", "",
                f"**Reference:** {q['reference_answer']}", "", "**Answer:**", "", run["answer"], "",
                "<details><summary>Retrieved passages</summary>", "", "```", evidence_of(run), "```", "",
                "</details>", ""]
    (EVAL / "grading_sheet.md").write_text("\n".join(out), encoding="utf-8")


def summarise(questions, retrieval, answers, verdicts) -> None:
    results = {}
    for q in questions:
        run, verdict = answers[q["id"]], verdicts[q["id"]]
        checks = code_checks(q, run)
        result = outcome(q, checks, verdict)
        results[q["id"]] = {"id": q["id"], "kind": q["kind"], "outcome": result,
                            "suggested_cause": suggested_cause(q, checks, result), **checks,
                            "judge_correct": verdict["correct"], "judge_supported": verdict["supported"],
                            "guardrail": run["guardrail"]["outcome"] if run["guardrail"] else "",
                            "model_calls": run["model_calls"], "tokens_in": run["tokens"]["input"],
                            "tokens_out": run["tokens"]["output"], "seconds": run["seconds"]}

    with (EVAL / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(next(iter(results.values())).keys()))
        writer.writeheader()
        writer.writerows(results.values())

    n = len(results)
    counts = {}
    for r in results.values():
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    causes = {}
    for r in results.values():
        if r["suggested_cause"]:
            causes[r["suggested_cause"]] = causes.get(r["suggested_cause"], 0) + 1
    mean = lambda key: round(sum(r[key] for r in results.values()) / n, 1)  # noqa: E731
    agree = agreement(verdicts)
    summary = {
        "questions": n,
        "by_kind": {k: sum(r["kind"] == k for r in results.values()) for k in ("single", "multi", "unanswerable")},
        "outcomes": counts,
        "code_checks": {
            "searched": sum(r["searched"] for r in results.values()),
            "citations_valid": sum(r["citations_valid"] for r in results.values()),
            "answers_without_citation": sum(not r["cites_something"] for r in results.values()),
            "retrieved_expected": sum(bool(r["retrieved_expected"]) for r in results.values()),
            "cited_expected": sum(bool(r["cited_expected"]) for r in results.values()),
            "guardrail_fired": sum(bool(r["guardrail"]) for r in results.values()),
        },
        "suggested_failure_causes": causes,
        "cost_per_question": {"agent_tokens_in": mean("tokens_in"), "agent_tokens_out": mean("tokens_out"),
                              "agent_seconds": mean("seconds"), "agent_model_calls": mean("model_calls"),
                              "judge_tokens": round(sum(v["tokens"] for v in verdicts.values()) / n, 1)},
        "judge_vs_human": agree,
        "retrieval": {d: retrieval[d]["hit_rate"] for d in index.DESIGNS} | {"best_design": retrieval["best_design"]},
    }
    (EVAL / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log(f"\nOutcomes ({n} questions): " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    cc = summary["code_checks"]
    log(f"Code checks: searched {cc['searched']}/{n}, citations valid {cc['citations_valid']}/{n}, "
        f"expected passage retrieved {cc['retrieved_expected']}, guardrail fired {cc['guardrail_fired']}")
    if causes:
        log("Suggested failure causes (check each by hand): " + ", ".join(f"{v} {k}" for k, v in causes.items()))
    c = summary["cost_per_question"]
    log(f"Cost per question: {c['agent_tokens_in'] + c['agent_tokens_out']:,.0f} agent tokens, "
        f"{c['agent_seconds']} s, {c['agent_model_calls']} model calls; judge {c['judge_tokens']:,.0f} tokens")
    if agree:
        log(f"Judge vs your grades ({agree['graded']} answers): correct {agree['agree_correct']:.0%}, "
            f"supported {agree['agree_supported']:.0%}, both {agree['agree_both']:.0%}")
    else:
        log("Judge vs your grades: grade 20+ answers in eval/human_labels.csv from eval/grading_sheet.md, "
            "then run again (answers and verdicts are reused, so it costs nothing).")
    log("Saved: eval/retrieval.json, answers.jsonl, verdicts.jsonl, results.csv, summary.json, grading_sheet.md")


async def main() -> None:
    questions = read_questions()
    log(f"{len(questions)} questions. Retrieval ...")
    retrieval = evaluate_retrieval(questions)
    answers = await run_agent(questions)
    write_grading_sheet(questions, answers)
    verdicts = await run_judge(questions, answers)
    summarise(questions, retrieval, answers, verdicts)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (index.QuotaExhausted, DailyQuota) as error:
        log(f"\n{error}")
        sys.exit(2)