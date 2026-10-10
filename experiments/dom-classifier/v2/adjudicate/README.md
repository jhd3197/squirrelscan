# Adjudicator, label model and confident learning

How Treetop turns several noisy label sources into one training label (#2307).
Rules and vocabulary are in [`../../DEVELOPMENT.md`](../../DEVELOPMENT.md). Real paths and
which run is live are in the private runbook.

## The flow

1. **Find suspect labels.** `confident_learning.py` flags train and val items
   whose label the out-of-fold model confidently contradicts, plus every Jev vs
   Clef page-type disagreement. `v82_candidates.py` adds pages whose v8 or v8.1
   label went stale under policy v8.2. `overnight_list.py` ranks them all into
   one capped list.
2. **Render what a human would see.**
   - `prepare-render.ts` reads the stored HTML and marks the section targets.
   - `modal_render.py` replays pages on Modal (`render-worker.ts` →
     `labeler/replay-capture.ts`, egress guard unchanged) and composes the
     views.
   - `reveal-capture.ts` / `reveal-worker.ts` force hidden targets visible and
     screenshot just the element.
   - `replay-html.ts` handles the scripts-off replay details.
3. **Adjudicate.** `adjudicate.py` holds the prompts, views and answer schema.
   Two transports:
   - `adjudicate.py` talks to the API directly;
   - `adjudicate_cli.py` runs one locked-down `claude -p` per item on the Team
     login.

   Each answer has `label`, `confidence`, `runner_up`, `render_quality` and
   `reason`. An invalid answer is an abstention, never a label.
4. **Combine.**
   - `build_votes.py` writes every source's vote per item in the trainers'
     label space.
   - `label_model.py` combines them into one posterior per item. Each source is
     weighted by its per-class accuracy on calibration gold.
5. **Check against humans.**
   - `calibration_sample.py` and `handoff.py` build the calibration-gold cohort
     (train/val only, never test gold).
   - `pilot_report.py` reports agreement while no gold exists.
   - `policy_rules.py` and `rule_candidates.py` hold the deterministic page-type
     rules. A candidate rule is applied only after calibration gold confirms it.

## The `claude -p` lockdown

Every item is one process with:
- no tools (`--tools ""`), so screenshots travel inline as image blocks;
- `--safe-mode` and `--restricted`;
- an empty MCP config;
- a replaced system prompt that treats page content as data;
- an empty working directory and no session persistence.

The init event of every run is checked: any tool beyond StructuredOutput, or
any MCP server, aborts the run (`lockdown_violated`). Never relax this to make
a run work.

The same rules apply to any other model runner (for example the Codex CLI):
- no shell or tools;
- read-only, from an empty directory;
- abort on any tool or command event in the stream.

## Usage guard

`adjudicate_cli.py` reads the five-hour window utilisation from each result.
It stops at `--stop-utilization`, on a non-`allowed` status, or on any limit
error, and never retries into the limit. Runs resume by `itemId`. The
start/stop thresholds are in the private runbook.

## Arms and modalities

- **Arm:** model × effort (`ARMS` in `adjudicate.py`).
- **Modality:**
  - `image`: page screenshot or element crop + context;
  - `image_scripts`: rendered with scripts on;
  - `revealed`: the forced-reveal element shot;
  - `text`: the packet only.

Record the arm, modality and `PROMPT_REVISION` on every output row. Bump
`PROMPT_REVISION` whenever the prompt or schema changes.
