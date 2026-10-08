"""Pack both heads into ONE artifact the TypeScript runtime can evaluate.

Bit-parity is achieved by construction rather than by hope: the thresholds are
rounded to float32 HERE, a reference evaluator in this file walks the rounded
trees in float32, and `predict.ts` implements the same walk. What is compared
in the parity test is TS against that reference, so the only thing that can
differ is the implementation, not the arithmetic.

The float32 rounding cost against LightGBM's own float64 predictions is
measured and reported rather than assumed to be zero.

Binary layout, little-endian, one block per head:

    magic   "SQDC"                    4 bytes
    version uint32 = 1
    numTrees uint32
    numClass uint32
    perTreeInternal uint32[numTrees]
    perTreeLeaves   uint32[numTrees]
    (pad to 4)
    feature   uint16[totalInternal]   which feature the node splits on
    (pad to 4)
    threshold float32[totalInternal]  compared with <=, left when true
    left      int32[totalInternal]    >= 0 internal index, < 0 leaf -(v)-1
    right     int32[totalInternal]
    leaf      float32[totalLeaves]
    flags     uint8[totalInternal]    version 2 only: missing-value handling

Version 2 exists for heads trained with missing values (page type with the
cross-page block dropped to NaN, #2307). A node's flags are LightGBM's own
missing semantics: bits 0-1 the missing type (0 None, 1 Zero, 2 NaN), bit 2
default_left. The walk is LightGBM's NumericalDecision: a NaN on a node whose
missing type is not NaN reads as 0.0; a NaN on a NaN node, or a zero (and so a
NaN) on a Zero node, goes the default direction; anything else compares <=.
A head with no missing-aware node is written as version 1, byte for byte what
it was before, so every earlier artifact and runtime is unaffected.

Tree t predicts for class t % numClass; the class score is the sum of its
trees' leaf values, and the probabilities are a softmax over those scores.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import struct
from pathlib import Path

import numpy as np

#: The dataset directory under --data-root. The 2026-09-19 build is `silver-v1`;
#: a larger corpus is built beside it rather than over it, and every script in
#: this directory has to agree on which one is being read.
DATASET = os.environ.get("DOM_CLASSIFIER_DATASET_VERSION", "silver-v1")


MAGIC = b"SQDC"
VERSION = 1
#: Tree blocks carrying per-node missing-value flags.
VERSION_MISSING = 2
MISSING_CODES = {None: 0, "None": 0, "Zero": 1, "NaN": 2}
DEFAULT_LEFT = 4
#: LightGBM's kZeroThreshold: |x| at or under it is "zero" for a Zero node.
ZERO_THRESHOLD = 1e-35


def flatten_tree(node: dict, columns: list[int] | None) -> tuple[list[dict], list[float]]:
    """LightGBM's nested tree -> flat internal nodes + leaf values.

    `columns` maps this head's training-matrix column back to its index in the
    FULL featurizer vector, and the mapping is baked in here. The runtime then
    indexes the feature row directly, with no per-node indirection and no
    branch on whether the head was trained on a subset.
    """
    internal: list[dict] = []
    leaves: list[float] = []

    def walk(n: dict) -> int:
        """Returns the child encoding: >= 0 internal index, < 0 leaf."""
        if "leaf_value" in n:
            leaves.append(float(n["leaf_value"]))
            return -len(leaves)  # -1 based, so leaf 0 encodes as -1
        # Anything other than a plain numeric "<=" split would change the
        # semantics the TS walk implements, so refuse rather than mis-evaluate.
        if n.get("decision_type") != "<=":
            raise ValueError(f"unsupported decision_type {n.get('decision_type')!r}")
        if n.get("missing_type") not in MISSING_CODES:
            raise ValueError(f"unsupported missing_type {n.get('missing_type')!r}")
        index = len(internal)
        internal.append({})
        left = walk(n["left_child"])
        right = walk(n["right_child"])
        split = int(n["split_feature"])
        missing = MISSING_CODES[n.get("missing_type")]
        internal[index] = {
            "feature": columns[split] if columns is not None else split,
            "threshold": float(n["threshold"]),
            "left": left,
            "right": right,
            # default_left only means something when a value can be missing.
            "flags": missing | (DEFAULT_LEFT if missing and n.get("default_left") else 0),
        }
        return index

    walk(node)
    return internal, leaves


def pack(trees: list[dict], num_class: int, columns: list[int] | None) -> tuple[bytes, dict]:
    per_internal: list[int] = []
    per_leaves: list[int] = []
    features: list[int] = []
    thresholds: list[float] = []
    lefts: list[int] = []
    rights: list[int] = []
    leaf_values: list[float] = []
    flags: list[int] = []

    for tree in trees:
        internal, leaves = flatten_tree(tree["tree_structure"], columns)
        offset_internal = len(features)
        offset_leaf = len(leaf_values)
        for node in internal:
            features.append(node["feature"])
            thresholds.append(node["threshold"])
            flags.append(node["flags"])
            # Re-base both child pointers into the concatenated arrays.
            lefts.append(
                node["left"] + offset_internal
                if node["left"] >= 0
                else node["left"] - offset_leaf
            )
            rights.append(
                node["right"] + offset_internal
                if node["right"] >= 0
                else node["right"] - offset_leaf
            )
        leaf_values.extend(leaves)
        per_internal.append(len(internal))
        per_leaves.append(len(leaves))

    if features and max(features) > 0xFFFF:
        raise ValueError("feature index does not fit in uint16")

    version = VERSION_MISSING if any(flags) else VERSION
    # A missing-value split can carry a threshold like 1e300 (every present
    # value goes left). In float32 that is +inf, and `x <= inf` decides every
    # finite x exactly as `x <= 1e300` does, so the overflow is the right
    # answer, not a lossy one: computed once here, without numpy's warning.
    with np.errstate(over="ignore"):
        threshold_f32 = np.asarray(thresholds, dtype="<f4")
    out = bytearray()
    out += MAGIC
    out += struct.pack("<III", version, len(trees), num_class)
    out += np.asarray(per_internal, dtype="<u4").tobytes()
    out += np.asarray(per_leaves, dtype="<u4").tobytes()
    out += np.asarray(features, dtype="<u2").tobytes()
    while len(out) % 4:
        out += b"\x00"
    out += threshold_f32.tobytes()
    out += np.asarray(lefts, dtype="<i4").tobytes()
    out += np.asarray(rights, dtype="<i4").tobytes()
    out += np.asarray(leaf_values, dtype="<f4").tobytes()
    if version == VERSION_MISSING:
        out += np.asarray(flags, dtype="<u1").tobytes()

    reference = {
        "feature": np.asarray(features, dtype=np.uint16),
        "threshold": threshold_f32.astype(np.float32),
        "left": np.asarray(lefts, dtype=np.int32),
        "right": np.asarray(rights, dtype=np.int32),
        "leaf": np.asarray(leaf_values, dtype=np.float32),
        "flags": np.asarray(flags, dtype=np.uint8) if version == VERSION_MISSING else None,
        "perInternal": np.asarray(per_internal, dtype=np.int64),
        "perLeaves": np.asarray(per_leaves, dtype=np.int64),
        "numClass": num_class,
        "numTrees": len(trees),
    }
    return bytes(out), reference


def evaluate(reference: dict, x: np.ndarray) -> np.ndarray:
    """The float32 reference evaluator. predict.ts implements exactly this."""
    n_rows = x.shape[0]
    num_class = reference["numClass"]
    scores = np.zeros((n_rows, num_class), dtype=np.float32)
    starts = np.concatenate([[0], np.cumsum(reference["perInternal"])])
    leaf_starts = np.concatenate([[0], np.cumsum(reference["perLeaves"])])
    feature = reference["feature"]
    threshold = reference["threshold"]
    left = reference["left"]
    right = reference["right"]
    leaf = reference["leaf"]
    flags = reference.get("flags")

    for row in range(n_rows):
        values = x[row]
        for tree in range(reference["numTrees"]):
            node = int(starts[tree])
            if reference["perInternal"][tree] == 0:
                # A stump: one leaf, no split. The child encoding cannot name
                # it, so it is addressed by this tree's own leaf offset -- NOT
                # leaf[0], which is the first tree's leaf.
                scores[row, tree % num_class] += leaf[int(leaf_starts[tree])]
                continue
            while True:
                value = np.float32(values[feature[node]])
                if flags is None:
                    nxt = left[node] if value <= threshold[node] else right[node]
                else:
                    nxt = missing_aware_step(value, int(flags[node]), threshold[node],
                                             left[node], right[node])
                if nxt < 0:
                    scores[row, tree % num_class] += leaf[-nxt - 1]
                    break
                node = int(nxt)
    exp = np.exp(scores.astype(np.float64) - scores.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def missing_aware_step(value: np.float32, flag: int, threshold: np.float32, left: int,
                       right: int) -> int:
    """LightGBM's NumericalDecision for one node (see the module docstring)."""
    missing = flag & 3
    if np.isnan(value) and missing != 2:
        value = np.float32(0.0)
    if (missing == 1 and abs(float(value)) <= ZERO_THRESHOLD) or (missing == 2 and np.isnan(value)):
        return left if flag & DEFAULT_LEFT else right
    return left if value <= threshold else right


def head(model_path: Path, columns_path: Path | None, classes: list[str], threshold: float,
         class_thresholds: list[float] | None = None, extra_names: list[str] | None = None):
    import lightgbm as lgb

    booster = lgb.Booster(model_file=str(model_path))
    dumped = booster.dump_model()
    columns = (
        np.load(columns_path).astype(np.int64).tolist() if columns_path is not None else None
    )
    blob, reference = pack(dumped["tree_info"], len(classes), columns)
    return booster, {
        "classes": classes,
        "featureColumns": columns,
        "confidenceThreshold": threshold,
        # Per class, chosen on val for precision >= 0.95 on the non-declared
        # population. A value above 1 means the class abstains: the model
        # cannot call it reliably, so it never fires.
        "classThresholds": class_thresholds,
        "numClass": len(classes),
        "numTrees": len(dumped["tree_info"]),
        # Page type with cross-page features: the names of the block appended
        # after [numeric, tokens], in order. The runtime fills it from the
        # caller's link graph, or with NaN when there is none.
        **({"extraFeatureNames": extra_names} if extra_names else {}),
        "trees": base64.b64encode(blob).decode("ascii"),
    }, reference


def pagetype_parity(args, schema: dict, booster, reference: dict, base_width: int,
                    extra_width: int, out: Path) -> dict:
    """Page-type parity on VAL pages, twice: with the cross-page block as
    computed from the link graph, and with that block MISSING (NaN), which is
    what the runtime passes when the caller has no graph. Both must keep every
    argmax LightGBM gives; the fixture lets predict.test.ts hold TS to the same."""
    import gzip

    dataset = Path(args.parity_dataset) if args.parity_dataset else Path(args.data_root) / DATASET
    extra = {}
    with open(args.pagetype_extra_features) as handle:
        for line in handle:
            row = json.loads(line)
            extra[row["p"]] = row["f"]
    n_numeric = schema["pageFeatureCount"]
    rows = []
    with gzip.open(dataset / "pages-val.jsonl.gz", "rt") as handle:
        for line in handle:
            page = json.loads(line)
            if page["p"] not in extra:
                continue
            vector = np.zeros(base_width + extra_width, dtype=np.float32)
            vector[:n_numeric] = page["f"]
            for bucket in page["k"]:
                vector[n_numeric + bucket] = 1.0
            vector[base_width:] = extra[page["p"]]
            rows.append(vector)
            if len(rows) >= args.pagetype_parity_rows:
                break
    with_graph = np.vstack(rows)
    no_graph = with_graph.copy()
    no_graph[:, base_width:] = np.nan
    report = {"rows": len(rows)}
    fixture = {"rows": len(rows), "width": int(with_graph.shape[1]), "baseWidth": base_width,
               "note": "sparse rows WITH the cross-page block; noGraph = same rows with "
                       "columns >= baseWidth set to NaN"}
    for name, matrix in (("graph", with_graph), ("noGraph", no_graph)):
        ref = evaluate(reference, matrix)
        lgb_probs = booster.predict(matrix)
        report[name] = {"maxAbsProbDiff": float(np.abs(ref - lgb_probs).max()),
                        "argmaxAgreement": float(np.mean(ref.argmax(1) == lgb_probs.argmax(1)))}
        if report[name]["argmaxAgreement"] < 1.0:
            raise SystemExit(f"page-type float32 packing changed an argmax ({name}); refusing")
        fixture[f"probs{name[0].upper()}{name[1:]}"] = [[float(v) for v in r] for r in ref]
    fixture["x"] = [{"i": np.flatnonzero(r).tolist(), "v": [float(r[j]) for j in np.flatnonzero(r)]}
                    for r in with_graph]
    (out / "parity-pagetype.json").write_text(json.dumps(fixture, separators=(",", ":")))
    print(f"page-type parity: {json.dumps(report)}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--parity-rows", type=int, default=10_000)
    parser.add_argument("--section-dir", default="",
                        help="trained section-purpose head (section-model.txt, "
                             "class-thresholds.json, sample/X-holdout.npz); optional third head")
    parser.add_argument("--pagetype-dir", default="pagetype",
                        help="subdirectory of --models holding the page-type head")
    parser.add_argument("--pagetype-extra-names", default="",
                        help="names.json of the cross-page block the page-type head was trained "
                             "with (--extra-features); its width must equal the head's extra width")
    parser.add_argument("--pagetype-extra-features", default="",
                        help="jsonl {p, f} of that block, for the page-type parity fixture")
    parser.add_argument("--parity-dataset", default="",
                        help="dataset dir whose pages-val.jsonl.gz feeds the page-type parity "
                             "fixture (default <data-root>/$DOM_CLASSIFIER_DATASET_VERSION)")
    parser.add_argument("--pagetype-parity-rows", type=int, default=2_000)
    args = parser.parse_args()

    models = Path(args.models)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    schema = json.loads((Path(args.data_root) / DATASET / "feature-schema.json").read_text())

    region_report = json.loads((models / "region" / "region-report.json").read_text())
    page_dir = models / args.pagetype_dir
    page_report_path = next(
        p for p in (page_dir / "pagetype-jev-report.json", page_dir / "pagetype-report.json")
        if p.exists()
    )
    page_report = json.loads(page_report_path.read_text())

    # The threshold is read off the VAL coverage table (holdout for a report
    # written before val had one): the LOWEST value at or above 0.5 whose
    # covered accuracy clears 0.98.
    #
    # The floor of 0.5 matters. Without it the rule picks 0.0, because accuracy
    # at full coverage is already 0.968 -- a bar the untruncated model clears
    # trivially, which would ship a threshold that never falls back to rules.
    # 0.98 is the bar that actually separates: it lands on 0.7, keeping 96.6%
    # of nodes on the model at 98.1% accuracy and macro-F1 0.82 on those,
    # and handing the remaining 3.4% to the deterministic path.
    coverage = region_report["shipped"].get(
        "coverageAtThresholdVal", region_report["shipped"]["coverageAtThreshold"]
    )
    region_threshold = next(
        (row["threshold"] for row in coverage
         if row["threshold"] >= 0.5 and row["accuracyOnCovered"] >= 0.98),
        0.7,
    )

    class_threshold_path = models / "region" / "class-thresholds.json"
    class_thresholds = (
        json.loads(class_threshold_path.read_text())["thresholds"]
        if class_threshold_path.exists()
        else None
    )
    region_booster, region_head, region_ref = head(
        models / "region" / "region-model.txt",
        models / "region" / "columns-model.npy",
        region_report["classes"],
        region_threshold,
        class_thresholds,
    )
    # Per-class page-type thresholds, chosen on val by the trainer, when it
    # wrote them (v8 does); older heads fall back to the single 0.7.
    page_threshold_path = page_dir / "class-thresholds.json"
    page_class_thresholds = (
        json.loads(page_threshold_path.read_text())["thresholds"]
        if page_threshold_path.exists()
        else None
    )
    extra_names = (json.loads(Path(args.pagetype_extra_names).read_text())
                   if args.pagetype_extra_names else None)
    page_booster, page_head, page_ref = head(
        page_dir / "pagetype-model.txt",
        None,
        page_report["classes"],
        0.7,
        page_class_thresholds,
        extra_names,
    )
    base_width = schema["pageFeatureCount"] + schema["tokenBuckets"]
    if page_booster.num_feature() != base_width + len(extra_names or []):
        raise SystemExit(f"page-type head has {page_booster.num_feature()} features, expected "
                         f"{base_width} + {len(extra_names or [])} extra")

    artifact = {
        "format": "squirrel-dom-classifier",
        "artifactVersion": 1,
        "featurizerVersion": schema["featurizerVersion"],
        "nodeFeatureCount": schema["nodeFeatureCount"],
        "pageFeatureCount": schema["pageFeatureCount"],
        "tokenBuckets": schema["tokenBuckets"],
        "region": region_head,
        "pageType": page_head,
        "pageTypeLabelSource": page_report.get("labelProvenance", "silver"),
    }
    section_booster = section_ref = None
    if args.section_dir:
        section_dir = Path(args.section_dir)
        section_thresholds = json.loads((section_dir / "class-thresholds.json").read_text())
        section_booster, artifact["section"], section_ref = head(
            section_dir / "section-model.txt",
            None,
            section_thresholds["classes"],
            1.01,
            section_thresholds["thresholds"],
        )
    artifact_path = out / "classifier-v2.json"
    artifact_path.write_text(json.dumps(artifact, separators=(",", ":")))
    size = artifact_path.stat().st_size
    print(f"artifact {size} bytes ({size / 1e6:.2f} MB)", flush=True)

    # Parity fixture: real holdout rows, the float32 reference probabilities,
    # and -- separately measured -- how far that reference sits from LightGBM.
    import scipy.sparse as sp

    x_hold = sp.load_npz(models / "region-sample" / "X-holdout.npz").tocsr()
    columns = np.load(models / "region" / "columns-model.npy")
    full = np.asarray(x_hold[: args.parity_rows].todense(), dtype=np.float32)
    # The packed trees index the FULL vector; LightGBM still wants the subset
    # it was trained on, so the two are fed differently on purpose.
    reference_probs = evaluate(region_ref, full)
    lgb_probs = region_booster.predict(full[:, columns])
    max_diff = float(np.abs(reference_probs - lgb_probs).max())
    argmax_agreement = float(
        np.mean(reference_probs.argmax(axis=1) == lgb_probs.argmax(axis=1))
    )
    print(f"float32 reference vs LightGBM: maxAbsProbDiff {max_diff:.3e}, "
          f"argmax agreement {argmax_agreement:.6f}", flush=True)
    if argmax_agreement < 1.0:
        raise SystemExit("float32 packing changed an argmax; refusing to ship this artifact")

    # Sparse rows: the matrix is ~10% dense, and a dense JSON fixture was 30 MB
    # of mostly the character "0".
    sparse_rows = []
    for row in full:
        nz = np.flatnonzero(row)
        # NOT rounded. A float32 written with 7 decimal places is a DIFFERENT
        # number, and a feature value that lands next to a split threshold then
        # takes the other branch in TS than it did here -- which showed up as a
        # 7.7e-5 probability gap on an otherwise perfect argmax match.
        # float(np.float32) is exact and round-trips through JSON.
        sparse_rows.append({"i": nz.tolist(), "v": [float(row[j]) for j in nz]})
    fixture = {
        "featurizerVersion": schema["featurizerVersion"],
        "rows": int(full.shape[0]),
        "width": int(full.shape[1]),
        "featureColumns": columns.tolist(),
        "note": "sparse rows over the FULL featurizer vector; the packed trees index it directly",
        "x": sparse_rows,
        "probs": [[float(v) for v in row] for row in reference_probs],
    }
    (out / "parity-region.json").write_text(json.dumps(fixture, separators=(",", ":")))

    section_parity = None
    if section_booster is not None:
        # Same fixture, for the section head: its rows are the section sample's
        # own holdout (a region-sample row is not necessarily a section row).
        xs = sp.load_npz(Path(args.section_dir) / "sample" / "X-holdout.npz").tocsr()
        full_s = np.asarray(xs[: args.parity_rows].todense(), dtype=np.float32)
        ref_s = evaluate(section_ref, full_s)
        lgb_s = section_booster.predict(full_s)
        section_parity = {
            "maxAbsProbDiff": float(np.abs(ref_s - lgb_s).max()),
            "argmaxAgreement": float(np.mean(ref_s.argmax(axis=1) == lgb_s.argmax(axis=1))),
            "rows": int(full_s.shape[0]),
        }
        if section_parity["argmaxAgreement"] < 1.0:
            raise SystemExit("float32 packing changed a section argmax; refusing to ship")
        rows_s = []
        for row in full_s:
            nz = np.flatnonzero(row)
            rows_s.append({"i": nz.tolist(), "v": [float(row[j]) for j in nz]})
        (out / "parity-section.json").write_text(json.dumps({
            "featurizerVersion": schema["featurizerVersion"], "rows": int(full_s.shape[0]),
            "width": int(full_s.shape[1]), "x": rows_s,
            "probs": [[float(v) for v in row] for row in ref_s],
        }, separators=(",", ":")))

    page_parity = None
    if extra_names:
        page_parity = pagetype_parity(args, schema, page_booster, page_ref, base_width,
                                      len(extra_names), out)

    (out / "export-report.json").write_text(
        json.dumps(
            {
                "artifactBytes": size,
                "artifactMB": round(size / 1e6, 3),
                "regionThreshold": region_threshold,
                "regionClassThresholds": class_thresholds,
                "regionTrees": region_head["numTrees"],
                "pageTypeTrees": page_head["numTrees"],
                "sectionTrees": artifact["section"]["numTrees"] if "section" in artifact else None,
                "sectionFloat32VsLightGBM": section_parity,
                "pageTypeFloat32VsLightGBM": page_parity,
                "pageTypeBlockVersion": struct.unpack("<I", base64.b64decode(
                    page_head["trees"])[4:8])[0],
                "float32VsLightGBM": {
                    "maxAbsProbDiff": max_diff,
                    "argmaxAgreement": argmax_agreement,
                    "rows": int(full.shape[0]),
                },
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
