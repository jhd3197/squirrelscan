"""Label queues and label-quality rules for the Common Crawl extension (#2307).

queue      Jev queue rows {pageId, contentHash, url, split} for every page
           of a dataset (page type v8.1 and section v3 read the same shape).
clef-queue the pages Jev answered with confidence < --max-conf, pool splits
           only (holdout labels never train anything), LOWEST confidence
           first, so a spend cap cuts the most certain pages, not a random set.
consensus  Jev vs Clef agreement (after the trainer's class merge) and the
           drop rule: a page whose Jev confidence is < --max-conf AND whose
           Clef vote disagrees has no usable label; it is left out of
           training AND of CV scoring, in both arms of the comparison.
drop-rows  section nodes whose v3 (region-level) label is a class the
           shipped head takes at tightest-node (v2) granularity: on new
           pages there is no v2 answer, so the v3 one would mix granularities.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path


def jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def cmd_queue(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    handles = {}
    counts: Counter = Counter()
    for row in jsonl(Path(args.dataset) / "page-index.jsonl.gz"):
        split = row["s"]
        if split not in handles:
            handles[split] = (out / f"queue-{split}.jsonl").open("w")
        handles[split].write(
            json.dumps(
                {"pageId": row["p"], "contentHash": row["p"], "url": row["u"], "split": split}
            )
            + "\n"
        )
        counts[split] += 1
    for handle in handles.values():
        handle.close()
    print(json.dumps(counts))


def cmd_clef_queue(args) -> None:
    urls = {r["p"]: r["u"] for r in jsonl(Path(args.dataset) / "page-index.jsonl.gz")}
    splits = set(args.splits.split(","))
    rows = []
    for path in args.jev:
        for r in jsonl(Path(path)):
            conf = r.get("confidence")
            if r["split"] in splits and conf is not None and conf < args.max_conf:
                rows.append((conf, r["pageId"], r["split"]))
    rows.sort()
    with open(args.out, "w") as handle:
        for conf, page, split in rows:
            handle.write(
                json.dumps(
                    {
                        "pageId": page,
                        "contentHash": page,
                        "url": urls[page],
                        "split": split,
                        "jevConfidence": conf,
                    }
                )
                + "\n"
            )
    print(json.dumps({"queued": len(rows), "bySplit": Counter(s for _, _, s in rows)}))


def cmd_consensus(args) -> None:
    merge = dict(pair.split(":") for pair in args.merge.split(",") if pair)
    fold = lambda label: merge.get(label, label)  # noqa: E731
    jev = {}
    for path in args.jev:
        for r in jsonl(Path(path)):
            jev[r["pageId"]] = r
    clef = {r["pageId"]: r for r in jsonl(Path(args.clef))}
    agree: Counter = Counter()
    seen: Counter = Counter()
    pairs: Counter = Counter()
    drop = []
    for page, c in clef.items():
        j = jev.get(page)
        if j is None:
            continue
        a, b = fold(j["pageType"]), fold(c["choice"])
        seen[a] += 1
        agree[a] += a == b
        if a != b:
            pairs[f"{a}->{b}"] += 1
            if (j.get("confidence") or 0) < args.max_conf:
                drop.append(page)
    Path(args.out).write_text("\n".join(drop) + "\n")
    report = {
        "compared": sum(seen.values()),
        "agree": sum(agree.values()),
        "agreement": round(sum(agree.values()) / max(sum(seen.values()), 1), 4),
        "perJevClass": {c: f"{agree[c]}/{seen[c]}" for c in sorted(seen)},
        "topDisagreements": pairs.most_common(15),
        "dropped": len(drop),
        "rule": f"drop when Jev confidence < {args.max_conf} and Clef (merged) disagrees",
    }
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


def cmd_drop_rows(args) -> None:
    classes = set(args.classes.split(","))
    n = 0
    with open(args.out, "w") as handle:
        for path in args.labels:
            for r in jsonl(Path(path)):
                if r.get("section") in classes:
                    handle.write(json.dumps({"pageId": r["pageId"], "nodeId": r["nodeId"]}) + "\n")
                    n += 1
    print(json.dumps({"dropRows": n}))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("queue")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("clef-queue")
    p.add_argument("--dataset", required=True)
    p.add_argument("--jev", nargs="+", required=True)
    p.add_argument("--max-conf", type=float, default=0.7)
    p.add_argument("--splits", default="train,val")
    p.add_argument("--out", required=True)
    p = sub.add_parser("consensus")
    p.add_argument("--jev", nargs="+", required=True)
    p.add_argument("--clef", required=True)
    p.add_argument("--merge", default="")
    p.add_argument("--max-conf", type=float, default=0.7)
    p.add_argument("--out", required=True)
    p.add_argument("--report", required=True)
    p = sub.add_parser("drop-rows")
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--classes", default="table_of_contents")
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    {
        "queue": cmd_queue,
        "clef-queue": cmd_clef_queue,
        "consensus": cmd_consensus,
        "drop-rows": cmd_drop_rows,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
