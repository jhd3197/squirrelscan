#!/usr/bin/env bun
/**
 * Registrable domain, dataset split and exclusion status for Common Crawl
 * candidate hosts (#2307 diversity extension). The split is computed HERE, by
 * the same rule build-dataset.ts applies (sha256(registrableDomain)[0:8] mod
 * 100, tldts with private suffixes), so the sampler never re-implements it in
 * another language and a candidate's planned split is the split it will land in.
 *
 *   bun run corpus/cc-domains.ts --in hosts.txt --exclusions <dir> --out domains.jsonl
 *
 * Input: one host or URL per line. Output: one JSONL row per input line,
 * {input, domain, domainHash, split, excluded}.
 */
import { createHash } from "node:crypto";
import { existsSync, readFileSync, writeFileSync } from "node:fs";

import { parse as parseDomain } from "tldts";

function argument(name: string): string {
  const index = Bun.argv.indexOf(name);
  const value = index < 0 ? undefined : Bun.argv[index + 1];
  if (!value || value.startsWith("--")) throw new Error(`${name} needs a value`);
  return value;
}

type Split = "train" | "val" | "holdout";

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

/** Identical to build-dataset.ts splitOf(). */
function splitOf(domain: string): Split {
  const bucket = parseInt(sha256(domain).slice(0, 8), 16) % 100;
  if (bucket < 80) return "train";
  if (bucket < 90) return "val";
  return "holdout";
}

function registrable(value: string): string {
  const input = /^https?:\/\//i.test(value) ? value : `https://${value}/`;
  const parsed = parseDomain(input, { allowPrivateDomains: true });
  return parsed.domain ?? parsed.hostname ?? "";
}

const exclusions = argument("--exclusions");
const lines = (path: string) =>
  existsSync(path) ? readFileSync(path, "utf8").split("\n").filter(Boolean) : [];
const excludedDomains = new Set(lines(`${exclusions}/domains.txt`));
const excludedHashes = new Set(lines(`${exclusions}/domain-hashes.txt`));
if (excludedDomains.size === 0) throw new Error(`no exclusions under ${exclusions}`);

const out: string[] = [];
for (const raw of lines(argument("--in"))) {
  const input = raw.trim();
  const domain = registrable(input.toLowerCase());
  const domainHash = domain ? sha256(domain).slice(0, 16) : "";
  out.push(
    JSON.stringify({
      input,
      domain,
      domainHash,
      split: domain ? splitOf(domain) : null,
      excluded: !domain || excludedDomains.has(domain) || excludedHashes.has(domainHash),
    }),
  );
}
writeFileSync(argument("--out"), `${out.join("\n")}\n`, { mode: 0o600 });
