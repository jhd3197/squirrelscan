#!/usr/bin/env bun
/**
 * Rebuild the page packets for a deterministic audit subset, for the reviewer.
 *
 * The label rows deliberately carry no page text, so the packet the reviewer
 * sees has to be rebuilt rather than copied. That is only safe if it is the
 * same packet: every rebuilt packet's sha256 is checked against the
 * `inputSha256` Jev recorded, and a mismatch fails the run rather than quietly
 * auditing a different page than the one that was labelled.
 */
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { createHash } from "node:crypto";
import { dirname, resolve } from "node:path";
import { gunzipSync } from "node:zlib";
import { Database } from "bun:sqlite";
import { loadPolicy, pageEvidenceFromHtml, serializePagePacket, sha256, POLICY_REVISIONS, type PolicyRevision } from "./jev-v5.ts";

const PRIVATE_ROOT = resolve(process.env.DOM_CLASSIFIER_DATA_ROOT ?? "");
if (!PRIVATE_ROOT) throw new Error("DOM_CLASSIFIER_DATA_ROOT must be set");
const privatePath = (raw: string) => {
  const path = resolve(raw);
  if (!path.startsWith(`${PRIVATE_ROOT}/`)) throw new Error(`path must be under ${PRIVATE_ROOT}`);
  return path;
};
function argument(name: string, fallback?: string): string {
  const index = Bun.argv.indexOf(name);
  if (index < 0) {
    if (fallback !== undefined) return fallback;
    throw new Error(`missing ${name}`);
  }
  const value = Bun.argv[index + 1];
  if (!value || value.startsWith("--")) throw new Error(`${name} needs a value`);
  return value;
}

const labelsPath = privatePath(argument("--labels"));
const queuePath = privatePath(argument("--queue"));
const outputPath = privatePath(argument("--output"));
const storePath = resolve(argument("--store"));
const target = Number(argument("--target", "300"));
const revision = argument("--policy-revision", "v7") as PolicyRevision;
if (!POLICY_REVISIONS.includes(revision)) throw new Error("bad --policy-revision");
const policy = loadPolicy(revision);

const labels = new Map(
  readFileSync(labelsPath, "utf8")
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line) as { pageId: string; inputSha256: string; split: string })
    .map((row) => [row.pageId, row]),
);
const queue = readFileSync(queuePath, "utf8")
  .split("\n")
  .filter(Boolean)
  .map((line) => JSON.parse(line) as { pageId: string; contentHash: string; url: string; split: string })
  .filter((row) => labels.has(row.pageId))
  // Deterministic and content-derived: the audit subset is reproducible from
  // the inputs, and independent of crawl or label order.
  .sort((left, right) =>
    createHash("sha256").update(left.pageId).digest("hex") <
    createHash("sha256").update(right.pageId).digest("hex")
      ? -1
      : 1,
  )
  .slice(0, target);

const store = new Database(storePath, { readonly: true });
const select = store.query<{ hash: string; content: Uint8Array }, [string, string]>(
  "select hash, content from content where hash >= ? and hash < ? limit 2",
);

const rows: string[] = [];
let mismatched = 0;
for (const row of queue) {
  const found = select.all(row.contentHash, `${row.contentHash}g`);
  if (found.length !== 1) continue;
  const bytes = found[0].content instanceof Uint8Array ? found[0].content : new Uint8Array(found[0].content);
  let html: string;
  try {
    html = new TextDecoder().decode(gunzipSync(bytes));
  } catch {
    html = new TextDecoder().decode(bytes);
  }
  const packet = serializePagePacket(pageEvidenceFromHtml(html, row.url, row.pageId));
  if (sha256(packet) !== labels.get(row.pageId)!.inputSha256) {
    mismatched++;
    continue;
  }
  rows.push(
    JSON.stringify(
      {
        packetRevision: policy.revision,
        recordType: "page",
        split: row.split,
        pageId: row.pageId,
        nodeId: null,
        captureHash: found[0].hash,
        stratum: "page",
        input: packet,
        inputSha256: sha256(packet),
        policySha256: policy.sha256,
        attributesAvailable: true,
      },
      [
        "attributesAvailable", "captureHash", "input", "inputSha256", "nodeId",
        "packetRevision", "pageId", "policySha256", "recordType", "split", "stratum",
      ],
    ),
  );
}
store.close();
if (mismatched)
  throw new Error(
    `${mismatched} rebuilt packets do not match the labelled bytes; the audit would not be of the labelled pages`,
  );
mkdirSync(dirname(outputPath), { recursive: true, mode: 0o700 });
writeFileSync(outputPath, rows.map((row) => `${row}\n`).join(""), { mode: 0o600 });
console.log(JSON.stringify({ audited: rows.length, requested: target, mismatched }));
