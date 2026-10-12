/**
 * One Common Crawl WARC record, fetched by byte range and unpacked (#2307).
 *
 * A CC locator is (warc_filename, offset, length): the byte range of one gzip
 * member in a WARC file on data.commoncrawl.org. Inside it: the WARC headers,
 * then the HTTP response (status line, headers, body). CC stores the body with
 * the transfer and content encodings already removed and renames the original
 * headers to `X-Crawler-*`; a plain `Content-Encoding` or chunked
 * `Transfer-Encoding` is still undone here, and counted, in case a record kept
 * one. The page's identity is sha256 of that body, as DomainGenius stores it.
 *
 * Page content is hostile: it is unpacked and hashed, never executed.
 */
import { createHash } from "node:crypto";
import { brotliDecompressSync, gunzipSync, inflateRawSync, inflateSync } from "node:zlib";

export const CC_BASE = "https://data.commoncrawl.org/";

export interface WarcResponse {
  warcType: string;
  targetUri: string;
  httpStatus: number | null;
  contentType: string | null;
  /** The HTTP body, encodings undone. */
  body: Uint8Array;
  /** sha256 hex of `body`. */
  sha256: string;
  /** Encodings this record still carried and that were undone here. */
  undone: string[];
  /** CC's WARC-Truncated reason (it caps stored payloads), or null. */
  truncated: string | null;
}

export class WarcError extends Error {}

const CRLF2 = [13, 10, 13, 10];

function indexOfSeq(bytes: Uint8Array, seq: number[], from = 0): number {
  outer: for (let i = from; i <= bytes.length - seq.length; i++) {
    for (let j = 0; j < seq.length; j++) if (bytes[i + j] !== seq[j]) continue outer;
    return i;
  }
  return -1;
}

function headerBlock(text: string): { first: string; headers: Map<string, string> } {
  const lines = text.split("\r\n");
  const headers = new Map<string, string>();
  for (const line of lines.slice(1)) {
    const colon = line.indexOf(":");
    if (colon > 0)
      headers.set(line.slice(0, colon).trim().toLowerCase(), line.slice(colon + 1).trim());
  }
  return { first: lines[0] ?? "", headers };
}

/** Undo `Transfer-Encoding: chunked`. */
export function dechunk(bytes: Uint8Array): Uint8Array {
  const parts: Uint8Array[] = [];
  let at = 0;
  while (at < bytes.length) {
    const lineEnd = indexOfSeq(bytes, [13, 10], at);
    if (lineEnd < 0) throw new WarcError("chunked body: no size line");
    const size = parseInt(new TextDecoder("latin1").decode(bytes.subarray(at, lineEnd)), 16);
    if (!Number.isFinite(size) || size < 0) throw new WarcError("chunked body: bad size");
    if (size === 0) break;
    const start = lineEnd + 2;
    if (start + size > bytes.length) throw new WarcError("chunked body: truncated");
    parts.push(bytes.subarray(start, start + size));
    at = start + size + 2;
  }
  return Buffer.concat(parts);
}

function undoContentEncoding(body: Uint8Array, encoding: string): Uint8Array {
  switch (encoding) {
    case "gzip":
    case "x-gzip":
      return gunzipSync(body);
    case "deflate":
      try {
        return inflateSync(body);
      } catch {
        return inflateRawSync(body);
      }
    case "br":
      return brotliDecompressSync(body);
    default:
      throw new WarcError(`content-encoding ${encoding}`);
  }
}

/** Unpack one gzip-member WARC record holding an HTTP response. */
export function parseWarcRecord(member: Uint8Array): WarcResponse {
  const raw = gunzipSync(member);
  const warcEnd = indexOfSeq(raw, CRLF2);
  if (warcEnd < 0) throw new WarcError("no WARC header end");
  const warc = headerBlock(new TextDecoder("latin1").decode(raw.subarray(0, warcEnd)));
  if (!warc.first.startsWith("WARC/")) throw new WarcError("not a WARC record");
  const warcType = warc.headers.get("warc-type") ?? "";
  const declared = Number(warc.headers.get("content-length") ?? "NaN");
  const blockStart = warcEnd + 4;
  const block = raw.subarray(
    blockStart,
    Number.isFinite(declared) ? blockStart + declared : raw.length,
  );
  if (warcType !== "response") throw new WarcError(`WARC-Type ${warcType || "missing"}`);

  const httpEnd = indexOfSeq(block, CRLF2);
  if (httpEnd < 0) throw new WarcError("no HTTP header end");
  const http = headerBlock(new TextDecoder("latin1").decode(block.subarray(0, httpEnd)));
  const status = /^HTTP\/[\d.]+\s+(\d{3})/.exec(http.first)?.[1];
  let body: Uint8Array = block.subarray(httpEnd + 4);
  const undone: string[] = [];
  if ((http.headers.get("transfer-encoding") ?? "").toLowerCase().includes("chunked")) {
    body = dechunk(body);
    undone.push("chunked");
  }
  const encoding = (http.headers.get("content-encoding") ?? "").trim().toLowerCase();
  if (encoding && encoding !== "identity") {
    body = undoContentEncoding(body, encoding);
    undone.push(encoding);
  }
  return {
    warcType,
    targetUri: warc.headers.get("warc-target-uri") ?? "",
    httpStatus: status ? Number(status) : null,
    contentType: http.headers.get("content-type") ?? null,
    body,
    sha256: createHash("sha256").update(body).digest("hex"),
    undone,
    truncated: warc.headers.get("warc-truncated") ?? null,
  };
}

export interface FetchOptions {
  /** Tries per range, the first included. */
  tries?: number;
  /** First backoff in ms; doubles per retry, capped at 60 s, and each wait is
   *  drawn from its upper half (jitter that still waits). */
  backoffMs?: number;
  /** Called on every throttle answer, for the run's circuit breaker. */
  onThrottle?: (status: number) => void;
  /** Awaited before every attempt, retries included: the run's rate limit. */
  beforeAttempt?: () => Promise<void>;
  fetchImpl?: typeof fetch;
}

/** Answers that mean "slow down", not "no such record". CC documents 503
 *  SlowDown; its CloudFront front also blocks an IP with 403 for minutes after
 *  ~2k range requests in a few minutes (2026-10-09, ~13 requests/s). */
export const THROTTLE_STATUSES: ReadonlySet<number> = new Set([403, 429, 503]);

/** Fetch one record's bytes by HTTP range, backing off on throttling. */
export async function fetchWarcRange(
  warcFilename: string,
  offset: number,
  length: number,
  options: FetchOptions = {},
): Promise<Uint8Array> {
  const tries = options.tries ?? 6;
  const doFetch = options.fetchImpl ?? fetch;
  const url = CC_BASE + warcFilename;
  let wait = options.backoffMs ?? 1000;
  let last = "";
  for (let attempt = 1; attempt <= tries; attempt++) {
    await options.beforeAttempt?.();
    try {
      const response = await doFetch(url, {
        headers: { Range: `bytes=${offset}-${offset + length - 1}` },
      });
      if (response.status === 206) {
        const bytes = new Uint8Array(await response.arrayBuffer());
        if (bytes.length !== length) throw new WarcError(`got ${bytes.length} of ${length} bytes`);
        return bytes;
      }
      last = `HTTP ${response.status}`;
      await response.body?.cancel();
      if (THROTTLE_STATUSES.has(response.status)) options.onThrottle?.(response.status);
      else if (response.status < 500) throw new WarcError(last);
    } catch (error) {
      if (error instanceof WarcError) throw error;
      last = String(error).slice(0, 200);
    }
    if (attempt < tries) {
      const cap = Math.min(wait, 60_000);
      await Bun.sleep(cap / 2 + (Math.random() * cap) / 2);
      wait *= 2;
    }
  }
  throw new WarcError(`gave up after ${tries} tries: ${last}`);
}
