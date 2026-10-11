# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Opus teacher over the gold sets: hard items, the human review queue, the report (#2307).

Opus 5.5 labels gold-v1, gold-cal-v1 and gold-v1b in place of the human; the
human reviews only what this file routes to him.

  hard     pass-2 selection, written to <set>/teacher/hard-ids.txt. An item is hard
           when pass 1 answered with confidence < 0.6, rated the render other than
           ok, or disagreed with every outside source (page: Jev and Clef where
           Clef voted; section: Jev). Pass 2 (effort high) runs on these only.
  queue    the review cohort teacher-review-v1:
             - hard items still unresolved after pass 2 (confidence < 0.6, the
               label changed between passes, or an unusable render), SHOWN with
               the pass-2 answer;
             - a random audit of the teacher, stratified by axis x platform and
               shown BLIND (no suggestion), so the audit measures Opus's own error
               rate; human answers already given count toward it and are not re-asked;
             - ~5% attention checks on shown items (a distant wrong class).
           Writes items.jsonl (the gold manifest shape), model-suggestions.jsonl
           (the labeller sidecar; blind items carry only their axis) and
           private/key.jsonl (Opus answers, outside sources, attention flags).
  report   per set: counts, window used, agreement of Opus with Jev, Clef and the
           human answers so far, per class; queue size and review minutes.

Outside sources never reach the prompt (adjudicate.py); they are compared here only.

    uv run --script teacher_queue.py hard --data <dom-classifier-data>
    uv run --script teacher_queue.py queue --data <dom-classifier-data> --audit 80
    uv run --script teacher_queue.py report --data <dom-classifier-data>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

SETS = ("gold-v1", "gold-cal-v1", "gold-v1b")
PASS1, PASS2 = "cli-opus-medium", "cli-opus-high"
HUMAN = "2026-09-18/human-ui-v1"
SEED = "teacher-review-v1"
AXIS_NAME = {"page": "pageTypeV8", "section": "sectionPurposeV3"}
PAGE_CLASSES = ["homepage", "about", "contact", "legal", "pricing", "blog_post", "article", "docs", "listing_index",
                "product_detail", "product_listing", "other"]  # fmt: skip
SECTION_CLASSES = ["faq_block", "reviews", "breadcrumbs", "business_info", "product_offer", "author_byline",
                   "video_embed", "ad", "cookie_consent", "newsletter_signup", "social_share", "related_content",
                   "comments", "promo_banner", "pagination", "table_of_contents", "account_login", "locale_switcher",
                   "none"]  # fmt: skip
# Seconds per question for a human (gold-v1's MORNING note: ~6 s page, 3-4 s section when shown).
SECONDS = {("page", False): 6, ("section", False): 4, ("page", True): 10, ("section", True): 7}


def jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def rank(key: str) -> str:
    return hashlib.sha256(f"{SEED}:{key}".encode()).hexdigest()


def load(data: Path) -> dict:
    """Every item of the three sets with its pass-1/pass-2 answers, outside sources and human answer."""
    human = human_answers(data)
    out = {}
    for name in SETS:
        teacher = data / name / "teacher"
        sources = {r["itemId"]: r for r in jsonl(teacher / "sources.jsonl")}
        p1 = {r["itemId"]: r for r in jsonl(teacher / "answers" / PASS1 / "answers-cli.jsonl") if not r.get("error")}
        p2 = {r["itemId"]: r for r in jsonl(teacher / "answers" / PASS2 / "answers-cli.jsonl") if not r.get("error")}
        manifest = {r["suggestionId"]: r for r in jsonl(data / name / "private" / "manifest.jsonl")}
        attention = {r["suggestionId"] for r in jsonl(data / name / "private" / "attention-key.jsonl")}
        for item in jsonl(teacher / "items.jsonl"):
            i = item["itemId"]
            out[i] = {
                **item,
                "sources": sources.get(i, {}),
                "pass1": p1.get(i),
                "pass2": p2.get(i),
                "human": human.get(i),
                "attentionInSet": i in attention,
                "manifest": manifest.get(i, {}),
            }
    return out


def human_answers(data: Path) -> dict[str, dict]:
    """The human's gold answers so far, keyed by suggestion id: label (or None = 'not OK' without a fix)."""
    out = {}
    for name, field in (("annotations.jsonl", "sectionPurposesV3"), ("page-annotations.jsonl", "pageTypesV8")):
        for r in jsonl(data / HUMAN / name):
            if not r.get("cohortId") or not r.get("modelSuggestionId"):
                continue
            labels = r.get(field) or []
            out[r["modelSuggestionId"]] = {
                "label": labels[0] if labels else None,
                "itemOf": r.get("sectionItemOfV3"),
                "review": r.get("modelReview"),
                "cohort": r["cohortId"],
                "at": r.get("timestamp"),
            }
    return out


def final(row: dict) -> dict | None:
    """The teacher's answer: pass 2 where it ran, else pass 1."""
    return row["pass2"] or row["pass1"]


def disagrees_with_sources(row: dict, answer: dict) -> bool:
    jev, clef = row["sources"].get("jev"), row["sources"].get("clef")
    if row["axis"] == "page":
        voted = [s for s in (jev, clef) if s]
        return bool(voted) and all(answer["label"] != s for s in voted)
    return bool(jev) and answer["label"] != jev


def is_hard(row: dict) -> list[str]:
    a = row["pass1"]
    if not a:
        return []
    reasons = []
    if a["confidence"] < 0.6:
        reasons.append("low_confidence")
    if a["renderQuality"] != "ok":
        reasons.append(f"render_{a['renderQuality']}")
    if disagrees_with_sources(row, a):
        reasons.append("disagrees_with_sources")
    return reasons


def unresolved(row: dict) -> list[str]:
    a, b = row["pass1"], row["pass2"]
    if not b:
        return []
    reasons = []
    if b["confidence"] < 0.6:
        reasons.append("low_confidence")
    if b["label"] != a["label"] or (b.get("itemOf") or None) != (a.get("itemOf") or None):
        reasons.append("passes_disagree")
    if b["renderQuality"] == "unusable":
        reasons.append("render_unusable")
    return reasons


def cmd_hard(args) -> None:
    rows = load(args.data)
    summary = {}
    for name in SETS:
        hard = {i: is_hard(r) for i, r in rows.items() if r["set"] == name}
        ids = [i for i, reasons in hard.items() if reasons]
        (args.data / name / "teacher" / "hard-ids.txt").write_text("".join(f"{i}\n" for i in ids))
        summary[name] = {"answered": sum(1 for r in rows.values() if r["set"] == name and r["pass1"]),
                         "hard": len(ids), "reasons": Counter(x for v in hard.values() for x in v)}  # fmt: skip
    print(json.dumps(summary, default=dict, indent=1))


def suggestion_id(key: str) -> str:
    h = rank(f"suggestion:{key}")
    return f"msug_{h[:8]}-{h[8:12]}-4{h[13:16]}-8{h[17:20]}-{h[20:32]}"


def distant(row: dict, classes: list[str]) -> str:
    """An attention-check class no source and no teacher pass gave."""
    taken = {row["sources"].get(k) for k in ("jev", "clef", "sonnet", "trainingLabel", "oofPred")}
    for a in (row["pass1"], row["pass2"]):
        if a:
            taken |= {a["label"], a.get("runnerUp"), a.get("itemOf")}
    pool = [c for c in classes if c not in taken and c != "none" and c != "other"]
    return sorted(pool, key=lambda c: rank(f"attention:{row['itemId']}:{c}"))[0]


def cmd_queue(args) -> None:
    rows = load(args.data)
    out_dir = args.data / "teacher-review-v1"
    (out_dir / "private").mkdir(parents=True, exist_ok=True)
    answered_by_human = {i for i, r in rows.items() if r["human"]}
    hard = {i: unresolved(r) for i, r in rows.items() if r["pass2"]}
    hard = {i: v for i, v in hard.items() if v and i not in answered_by_human}
    # Audit: blind, stratified by axis x platform, proportional to the cell's share of all items.
    want = max(0, args.audit - len(answered_by_human))
    eligible = [r for i, r in rows.items() if final(r) and i not in answered_by_human]
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in sorted(eligible, key=lambda r: rank(f"audit:{r['itemId']}")):
        cells[(r["axis"], r["platform"])].append(r)
    total = len(eligible)
    quota = {c: round(want * len(v) / total) for c, v in cells.items()}
    # Fix rounding so the audit is exactly `want`, adjusting the largest cells.
    order = sorted(cells, key=lambda c: -len(cells[c]))
    k = 0
    while sum(quota.values()) != want and order:
        c = order[k % len(order)]
        quota[c] += 1 if sum(quota.values()) < want else -1 if quota[c] > 0 else 0
        k += 1
    audit = {r["itemId"] for c, v in cells.items() for r in v[: quota[c]]}
    queue_ids = sorted(set(hard) | audit, key=lambda i: rank(f"order:{i}"))
    shown_ids = [i for i in queue_ids if i not in audit]
    n_checks = round(args.check_share * len(shown_ids))
    checks = set(sorted(shown_ids, key=lambda i: rank(f"check:{i}"))[:n_checks])
    # The labeller keeps a sidecar row only with provider "typesafe" (its sidecar schema, not who answered:
    # modelId says that) and a createdAt Date.parse accepts; anything else is dropped silently.
    created = datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    items, sidecar, key = [], [], []
    for position, i in enumerate(queue_ids):
        r = rows[i]
        a = final(r)
        classes = PAGE_CLASSES if r["axis"] == "page" else SECTION_CLASSES
        blind = i in audit
        shown = None if blind else (distant(r, classes) if i in checks else a["label"])
        offered = [a.get("itemOf"), a.get("runnerUp"), r["sources"].get("jev"), r["sources"].get("clef"), a["label"]]
        alternatives = [] if blind else [c for c in dict.fromkeys(offered) if c and c != shown and c in classes][:4]
        capture = json.loads((args.data / r["set"] / "captures" / f"{r['pageId']}.json").read_text())
        sid = suggestion_id(f"{r['set']}:{i}")
        axis = AXIS_NAME[r["axis"]]
        m = r["manifest"]
        items.append(
            {
                "suggestionId": sid,
                "position": position,
                "axis": axis,
                "blind": blind,
                "shown": shown,
                "alternatives": alternatives,
                "capturePageId": r["pageId"],
                "captureNodeId": r["nodeId"],
                "captureDir": f"{r['set']}/captures",
                "stratum": "audit" if blind else "hard",
                "sourceCohort": r["set"],
                "sourceSuggestionId": i,
                "slicePageId": r.get("slicePageId"),
                "featurizerNode": r.get("featurizerNode"),
                "url": r["url"],
                "platform": r["platform"],
                "lang": r.get("lang"),
                "split": m.get("split"),
            }
        )
        row = {
            "schemaVersion": 1,
            "id": sid,
            "pageId": r["pageId"],
            "nodeId": r["nodeId"],
            "captureHash": capture["captureHash"],
            "provider": "typesafe",
            "modelId": "teacher-opus-5-5",
            "modelRevision": a.get("promptRevision", "adjudicator-v3"),
            "promptRevision": a.get("promptRevision", "adjudicator-v3"),
            "snapshotHash": "sha256:" + hashlib.sha256(f"{sid}:{axis}:{shown}".encode()).hexdigest(),
            "rawAnswers": None,
            "provisional": True,
            "createdAt": created,
        }
        if blind:
            row.update({"mappedLabels": {}, "blindAxis": axis})
        else:
            row.update({"mappedLabels": {axis: [shown]}, "alternatives": {axis: alternatives}})
        sidecar.append(row)
        key.append(
            {
                "suggestionId": sid,
                "sourceCohort": r["set"],
                "sourceSuggestionId": i,
                "stratum": "audit" if blind else "hard",
                "hardReasons": hard.get(i, []),
                "attention": i in checks,
                "shownWrong": shown if i in checks else None,
                "teacher": {
                    "pass1": r["pass1"]
                    and {k: r["pass1"].get(k) for k in ("label", "itemOf", "confidence", "renderQuality", "reason")},
                    "pass2": r["pass2"]
                    and {k: r["pass2"].get(k) for k in ("label", "itemOf", "confidence", "renderQuality", "reason")},
                },  # fmt: skip
                "sources": r["sources"],
            }
        )
    (out_dir / "items.jsonl").write_text("".join(json.dumps(x) + "\n" for x in items))
    (out_dir / "model-suggestions.jsonl").write_text("".join(json.dumps(x) + "\n" for x in sidecar))
    (out_dir / "private" / "key.jsonl").write_text("".join(json.dumps(x) + "\n" for x in key))
    minutes = sum(SECONDS[(rows[x["sourceSuggestionId"]]["axis"], x["blind"])] for x in items) / 60
    print(json.dumps({"items": len(items), "hard": len(set(hard) - audit), "audit": len(audit),
                      "auditFromHumanAnswers": len(answered_by_human), "attentionChecks": len(checks),
                      "byAxis": Counter(x["axis"] for x in items), "bySet": Counter(x["sourceCohort"] for x in items),
                      "minutes": round(minutes)}, default=dict, indent=1))  # fmt: skip


def agreement(pairs) -> dict:
    pairs = [(a, b) for a, b in pairs if a and b]
    return {"n": len(pairs), "agree": round(sum(a == b for a, b in pairs) / len(pairs), 3) if pairs else None}


def cmd_report(args) -> None:
    rows = load(args.data)
    report = {}
    for name in SETS:
        rs = [r for r in rows.values() if r["set"] == name]
        answers = [
            jsonl(args.data / name / "teacher" / "answers" / arm / "answers-cli.jsonl") for arm in (PASS1, PASS2)
        ]
        set_report = {
            "items": len(rs),
            "pass1": sum(1 for r in rs if r["pass1"]),
            "pass2": sum(1 for r in rs if r["pass2"]),
            "errors": sum(1 for a in answers for x in a if x.get("error")),
            "listUsd": round(sum(x.get("usd", 0) for a in answers for x in a), 2),
            "windowSeen": [
                min((x["utilization"] for a in answers for x in a if x.get("utilization") is not None), default=None),
                max((x["utilization"] for a in answers for x in a if x.get("utilization") is not None), default=None),
            ],  # fmt: skip
        }
        for axis in ("page", "section"):
            ax = [r for r in rs if r["axis"] == axis and final(r)]
            per = {}
            for src in ("jev", "clef", "sonnet", "human"):

                def theirs(r, src=src):
                    if src == "human":
                        h = r["human"]
                        return h and (h["label"] if not h.get("itemOf") else "none")
                    return r["sources"].get(src)

                pairs = [(final(r)["label"], theirs(r)) for r in ax]
                per[src] = agreement(pairs)
                classes = sorted({b for _, b in pairs if b})
                per[f"{src}ByClass"] = {c: agreement([(a, b) for a, b in pairs if b == c]) for c in classes}
            per["itemOf"] = Counter(final(r).get("itemOf") for r in ax if final(r).get("itemOf"))
            per["renderQuality"] = Counter(final(r)["renderQuality"] for r in ax)
            set_report[axis] = per
        report[name] = set_report
    (args.data / "teacher-review-v1").mkdir(exist_ok=True)
    (args.data / "teacher-review-v1" / "report.json").write_text(json.dumps(report, indent=1, default=dict))
    for name, r in report.items():
        print(name, {k: r[k] for k in ("items", "pass1", "pass2", "errors", "listUsd", "windowSeen")})
        for axis in ("page", "section"):
            a = r[axis]
            print(
                f"  {axis}: jev {a['jev']} clef {a['clef']} sonnet {a['sonnet']} human {a['human']} itemOf {dict(a['itemOf'])}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, func in (("hard", cmd_hard), ("queue", cmd_queue), ("report", cmd_report)):
        p = sub.add_parser(name)
        p.add_argument("--data", type=Path, required=True)
        if name == "queue":
            p.add_argument("--audit", type=int, default=80)
            p.add_argument("--check-share", type=float, default=0.05)
        p.set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
