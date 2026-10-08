# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Pilot report for the visual adjudicator (#2307): agreement, cost, latency.

No gold exists yet, so this reports AGREEMENT, not accuracy: each adjudicator
arm against the training label (Jev-derived), the OOF model, Clef where it
voted, the free sources, and the other arms. Agreement is split by
confident-learning stratum: on `rest` items (OOF and label agree) a sound
adjudicator should mostly agree; on `cand` items (OOF contradicts the label)
disagreement is the point, and which side it takes is the interesting number.

    uv run --script pilot_report.py --items cohort.jsonl --votes votes.jsonl --answers <adjudicate dir> --out report.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def rate(pairs):
    pairs = [p for p in pairs if p[1] is not None]
    return {"n": len(pairs), "agree": round(sum(a == b for a, b in pairs) / len(pairs), 3) if pairs else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--votes", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    items = {i["itemId"]: i for i in jsonl(args.items)}

    def key_of(i: dict) -> str:
        return i["pageId"] if i["axis"] == "page" else f"{i['pageId']}#{i['nodeId']}"

    votes = defaultdict(dict)
    for v in jsonl(args.votes):
        votes[v["key"]][v["source"]] = v["label"]
    arms = {}
    for path in sorted(args.answers.glob("*/answers*.jsonl")):
        arm = path.parent.name
        mode = path.stem.split("-", 1)[1] if "-" in path.stem else "batch"
        arms[f"{arm}/{mode}"] = {r["itemId"]: r for r in jsonl(path)}
    report = {"arms": {}}
    for name, answers in arms.items():
        ok = {k: a for k, a in answers.items() if not a.get("error") and k in items}
        entry = {
            "answered": len(ok),
            "errors": dict(Counter(a["error"] for a in answers.values() if a.get("error"))),
            "usd": round(sum(a.get("usd", 0) for a in answers.values()), 4),
        }
        for axis in ("page", "section"):
            rows = [(items[k], a) for k, a in ok.items() if items[k]["axis"] == axis]
            if not rows:
                continue
            ax = {
                "n": len(rows),
                "usdPerItem": round(sum(a["usd"] for _, a in rows) / len(rows), 5),
                "tokensPerItem": {
                    t: round(statistics.mean(a["usage"][t] for _, a in rows))
                    for t in ("input", "cacheRead", "cacheWrite", "output")
                },
                "renderQuality": dict(Counter(a["renderQuality"] for _, a in rows)),
                "meanConfidence": round(statistics.mean(a["confidence"] for _, a in rows), 3),
            }
            lat = [a["latencyMs"] for _, a in rows if a.get("latencyMs")]
            if lat:
                ax["latencyMs"] = {
                    "p50": round(statistics.median(lat)),
                    "p90": round(sorted(lat)[int(0.9 * (len(lat) - 1))]),
                }
            batch = {a.get("batchSeconds") for _, a in rows if a.get("batchSeconds")}
            if batch:
                ax["batchWallSeconds"] = sorted(batch)
            ax["vsTrainingLabel"] = {
                s: rate([(a["label"], i["given"]) for i, a in rows if i["stratum"].endswith(s)])
                for s in ("|cand", "|rest")
            }
            ax["vsTrainingLabel"]["all"] = rate([(a["label"], i["given"]) for i, a in rows])
            ax["vsOOF"] = {
                s: rate([(a["label"], i.get("oofPred")) for i, a in rows if i["stratum"].endswith(s)])
                for s in ("|cand", "|rest")
            }
            # On candidates: does the adjudicator side with the label, the OOF model, or neither?
            sides = Counter()
            for i, a in rows:
                if i["stratum"].endswith("|cand") and i.get("oofPred") and i["oofPred"] != i["given"]:
                    sides[
                        "label" if a["label"] == i["given"] else "oof" if a["label"] == i["oofPred"] else "neither"
                    ] += 1
            ax["candidateSide"] = dict(sides)
            sources = Counter(s for i, _ in rows for s in votes.get(key_of(i), {}))
            ax["vsSources"] = {
                s: rate([(a["label"], votes.get(key_of(i), {}).get(s)) for i, a in rows])
                for s in sorted(sources)
                if not s.startswith("adj:")
            }
            ax["vsTrainingLabelByClass"] = {
                c: rate([(a["label"], i["given"]) for i, a in rows if i["given"] == c])
                for c in sorted({i["given"] for i, _ in rows})
            }
            ax["vsTrainingLabelByPlatform"] = {
                p: rate([(a["label"], i["given"]) for i, a in rows if i["platform"] == p])
                for p in sorted({i["platform"] for i, _ in rows})
            }
            ax["vsTrainingLabelByLang"] = {
                g: rate([(a["label"], i["given"]) for i, a in rows if i["lang"] == g])
                for g in sorted({i["lang"] for i, _ in rows})
            }
            ax["topDisagreements"] = dict(
                Counter(f"{i['given']}->{a['label']}" for i, a in rows if a["label"] != i["given"]).most_common(12)
            )
            entry[axis] = ax
        report["arms"][name] = entry
    # Arm-vs-arm agreement on shared items.
    names = sorted(arms)
    report["armPairs"] = {}
    for x in range(len(names)):
        for y in range(x + 1, len(names)):
            a, b = arms[names[x]], arms[names[y]]
            shared = [k for k in a if k in b and not a[k].get("error") and not b[k].get("error")]
            for axis in ("page", "section"):
                ks = [k for k in shared if items.get(k, {}).get("axis") == axis]
                if ks:
                    report["armPairs"][f"{names[x]} ~ {names[y]} ({axis})"] = rate(
                        [(a[k]["label"], b[k]["label"]) for k in ks]
                    )
    args.out.write_text(json.dumps(report, indent=1))
    for name, entry in report["arms"].items():
        for axis in ("page", "section"):
            if axis in entry:
                e = entry[axis]
                print(
                    f"{name:28s} {axis:7s} n={e['n']:3d} $/item={e['usdPerItem']:.5f} "
                    f"label(all/cand/rest)={e['vsTrainingLabel']['all']['agree']}/{e['vsTrainingLabel']['|cand']['agree']}/{e['vsTrainingLabel']['|rest']['agree']} "
                    f"side={e['candidateSide']} lat={e.get('latencyMs')} rq={e['renderQuality']}"
                )
    for pair, r in report["armPairs"].items():
        print(f"{pair}: {r}")


if __name__ == "__main__":
    main()
