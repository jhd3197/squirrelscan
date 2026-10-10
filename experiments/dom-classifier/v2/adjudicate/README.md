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
     views. Its workers pass `--disable-chromium-sandbox`: the container runs
     as root, where Chromium refuses its sandbox, and is itself the isolation.
     Run anywhere else, the workers keep the sandbox.
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

## Teacher runs

A teacher is a frontier model that labels gold items in the human's place;
the human reviews only what the teachers leave open.

- **Inputs:** `teacher_prep.py` cuts the views from a gold set's existing
  labeller captures with Pillow (`views.py`, the same crops the Modal renderer
  makes): no browser, no re-render. `teacher-packets.ts` builds the text
  packets from the capture's node walk. Outside labels (Jev, Clef, earlier
  arms) go to a separate `sources.jsonl` and never reach a prompt.
- **Section scope:** a section answer carries `scope: whole | item`, the
  labeller's "item of" answer. `item` means the outline is one or several
  items of that section (one comment, one FAQ question, the phone and email
  lines of a contact block). It maps to training label `none`, with the
  purpose kept as `itemOf`.
- **Second teacher (Codex CLI):** `adjudicate_codex.py` sends the same views,
  policy prompt and JSON schema to an OpenAI model.
  - Call the binary directly. A shell alias may add a full-access sandbox.
  - Flags: `--sandbox read-only --ephemeral --ignore-user-config --ignore-rules`,
    run from an empty directory.
  - The policy prompt replaces Codex's base instructions (`-c model_instructions_file=…`).
  - Shell, exec, browser, computer-use, app, plugin, skill, hook and
    image-generation features are disabled.
  - `-i` takes several values, so the prompt goes on stdin.
  - The `--json` stream is the guard: any item other than the agent message or
    its reasoning aborts the whole run.
  - The stream carries no model id. An id the account cannot use fails with a
    400, so the requested id is the one that ran.
- **Comparison and review:** `teacher_check.py` puts two teachers side by side
  with the outside sources, and emits the items where the teachers disagree as
  a BLIND labeller cohort, with a few agreed controls. `teacher_queue.py`
  selects the hard items for a second, higher-effort pass and builds a review
  queue: unresolved hard items are shown, a stratified audit is blind, and
  attention checks are added.

