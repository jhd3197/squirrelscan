# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Per-link section export for DomainGenius (#2307): one line per located link,
in the shape DG asked for (agent-messages treetop-dg A16), plus per-page counts.

    uv run --script export_link_sections.py --links <probe run>/links.jsonl --out <dir>

`<dir>/links.jsonl`: {page_sha256, i, href, section, region, rel, conf, evidence, source}
  section: the comments rule's label (`comments` / `reviews`) when it fires, else the
           section head's label when it answered, else the link's region label;
  region:  the region label on its own (`unknown` when the model abstained);
  conf:    1.0 for the rule, else the head's confidence (null when nothing answered);
  evidence: the rule's matched id/class token, digits folded (no page text);
  source:  `rule` (the comments rule, or a region read from markup such as <footer>)
           | `model` | null (nothing answered).
`<dir>/pages.jsonl`: {page_sha256, n_links_total, n_links_by_section}

Output is PRIVATE (hrefs). Only located links are written; unmatched manifest
links are counted in pages.jsonl as `unlocated`.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def link_row(r: dict) -> dict:
    if r.get("commentsRule"):
        section, conf, evidence, source = r["commentsRule"], 1.0, r.get("commentsRuleToken"), "rule"
    elif r.get("section"):
        section, conf, evidence, source = r["section"], r.get("sectionConfidence"), None, "model"
    elif r.get("region") and r["region"] != "unknown":
        section, conf, evidence, source = r["region"], r.get("regionConfidence"), None, r.get("regionSource")
    else:
        section, conf, evidence, source = "unknown", None, None, None
    return {
        "page_sha256": r["sha256"],
        "i": r["i"],
        "href": r["href"],
        "section": section,
        "region": r.get("region", "unknown"),
        "rel": r.get("rel", []),
        "conf": round(conf, 4) if isinstance(conf, float) else conf,
        "evidence": evidence,
        # The region head's markup answers (`rules`) are deterministic too.
        "source": "rule" if source == "rules" else source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--links", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    pages: dict[str, Counter[str]] = defaultdict(Counter)
    written = 0
    with (args.out / "links.jsonl").open("w") as handle:
        for line in args.links.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("matched"):
                pages[r["sha256"]]["unlocated"] += 1
                continue
            row = link_row(r)
            pages[r["sha256"]][row["section"]] += 1
            handle.write(json.dumps(row) + "\n")
            written += 1
    with (args.out / "pages.jsonl").open("w") as handle:
        for sha, counts in sorted(pages.items()):
            located = sum(n for k, n in counts.items() if k != "unlocated")
            handle.write(
                json.dumps({"page_sha256": sha, "n_links_total": located, "n_links_by_section": dict(counts)}) + "\n"
            )
    print(json.dumps({"links": written, "pages": len(pages)}))


if __name__ == "__main__":
    main()
