// The CC link batch (#2307): WARC unpacking, polite fetching, capture
// grouping, the route guard and DG's A16 rows. No network: fetch is faked.

import { describe, expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { gzipSync } from "node:zlib";

import {
  a16Row,
  bucketOfHash,
  captureKey,
  domainHashOf,
  groupCaptures,
  routeRefusal,
  type Guard,
  type LocatorRow,
} from "./cc-link-batch.ts";
import { dechunk, fetchWarcRange, parseWarcRecord, WarcError } from "./cc-warc.ts";
import type { LinkPosition } from "./link-position.ts";

const enc = new TextEncoder();

function record(httpHead: string, body: Uint8Array, warcType = "response", extra = ""): Uint8Array {
  const block = Buffer.concat([enc.encode(`${httpHead}\r\n\r\n`), body]);
  const warc = `WARC/1.0\r\nWARC-Type: ${warcType}\r\nWARC-Target-URI: https://a.example.org/p\r\n${extra}Content-Length: ${block.length}\r\n\r\n`;
  return gzipSync(Buffer.concat([enc.encode(warc), block, enc.encode("\r\n\r\n")]));
}

describe("parseWarcRecord", () => {
  test("a plain response: status, type, body and its sha256", () => {
    const body = enc.encode("<html><body>hi</body></html>");
    const rec = parseWarcRecord(
      record(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=UTF-8\r\nX-Crawler-Content-Encoding: gzip",
        body,
      ),
    );
    expect(rec.httpStatus).toBe(200);
    expect(rec.contentType).toBe("text/html; charset=UTF-8");
    expect(new TextDecoder().decode(rec.body)).toBe("<html><body>hi</body></html>");
    expect(rec.sha256).toBe(createHash("sha256").update(body).digest("hex"));
    expect(rec.undone).toEqual([]);
    expect(rec.truncated).toBeNull();
  });

  test("a kept chunked gzip body is undone, and said so", () => {
    const html = enc.encode("<p>x</p>");
    const gz = gzipSync(html);
    const chunked = Buffer.concat([
      enc.encode(`${gz.length.toString(16)}\r\n`),
      gz,
      enc.encode("\r\n0\r\n\r\n"),
    ]);
    const rec = parseWarcRecord(
      record(
        "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Encoding: gzip",
        chunked,
        "response",
        "WARC-Truncated: length\r\n",
      ),
    );
    expect(new TextDecoder().decode(rec.body)).toBe("<p>x</p>");
    expect(rec.undone).toEqual(["chunked", "gzip"]);
    expect(rec.truncated).toBe("length");
  });

  test("a request record or garbage is a WarcError", () => {
    expect(() => parseWarcRecord(record("GET / HTTP/1.1", new Uint8Array(), "request"))).toThrow(
      WarcError,
    );
    expect(() => parseWarcRecord(gzipSync(enc.encode("not a warc")))).toThrow(WarcError);
    expect(() => dechunk(enc.encode("zz\r\n"))).toThrow(WarcError);
  });
});

describe("fetchWarcRange", () => {
  const ok = (bytes: Uint8Array) => new Response(Buffer.from(bytes), { status: 206 });

  test("asks for the exact range, backs off on 503, and reports the throttle", async () => {
    const asked: string[] = [];
    let calls = 0;
    let throttled = 0;
    const fake = (async (_url: string, init?: RequestInit) => {
      asked.push(new Headers(init?.headers).get("range") ?? "");
      return ++calls === 1 ? new Response("slow down", { status: 503 }) : ok(new Uint8Array(10));
    }) as unknown as typeof fetch;
    const bytes = await fetchWarcRange("crawl-data/x.warc.gz", 100, 10, {
      fetchImpl: fake,
      backoffMs: 1,
      onThrottle: () => throttled++,
    });
    expect(bytes.length).toBe(10);
    expect(asked).toEqual(["bytes=100-109", "bytes=100-109"]);
    expect(throttled).toBe(1);
  });

  test("a 404 or a short read fails at once; endless 503s give up", async () => {
    const answer = (status: number, size = 0) =>
      (async () =>
        status === 206
          ? ok(new Uint8Array(size))
          : new Response("", { status })) as unknown as typeof fetch;
    await expect(fetchWarcRange("f", 0, 10, { fetchImpl: answer(404) })).rejects.toThrow(
      "HTTP 404",
    );
    await expect(fetchWarcRange("f", 0, 10, { fetchImpl: answer(206, 4) })).rejects.toThrow(
      "got 4 of 10",
    );
    await expect(
      fetchWarcRange("f", 0, 10, { fetchImpl: answer(503), tries: 3, backoffMs: 1 }),
    ).rejects.toThrow("gave up after 3 tries");
  });
});

const row = (over: Partial<LocatorRow> = {}): LocatorRow => {
  const url = over.source_url ?? "https://www.a.example.org/post";
  const hash = domainHashOf(url)!;
  return {
    locator_id: "cc-1",
    crawl: "CC-MAIN-2024-10",
    warc_filename: "crawl-data/a.warc.gz",
    offset: 10,
    length: 20,
    source_url: url,
    domain_hash: hash,
    bucket: bucketOfHash(hash),
    sealed: false,
    links: [{ href: "https://t.example.com/", anchor_text: "t" }],
    ...over,
  };
};

describe("groupCaptures", () => {
  test("one capture per byte range; rows naming it merge links and ids", () => {
    const groups = groupCaptures([
      row(),
      row({
        locator_id: "cc-2",
        links: [
          { href: "https://t.example.com/", anchor_text: "t" },
          { href: "https://u.example.com/", anchor_text: "u" },
        ],
      }),
      row({ offset: 99 }),
    ]);
    expect(groups.length).toBe(2);
    expect(groups[0]!.locatorIds).toEqual(["cc-1", "cc-2"]);
    expect(groups[0]!.links!.map((l) => l.href)).toEqual([
      "https://t.example.com/",
      "https://u.example.com/",
    ]);
    expect(groups[0]!.key).toBe(captureKey(row()));
  });

  test("an empty link list means every outbound link, and wins the merge", () => {
    expect(groupCaptures([row(), row({ links: [] })])[0]!.links).toBeNull();
  });
});

describe("routeRefusal", () => {
  const guard = (route: Guard["route"], gold: string[] = []): Guard => ({
    route,
    sealed: [],
    gold,
  });
  // A domain in each half of the split, found deterministically.
  const domainWhere = (want: (bucket: number) => boolean): string => {
    for (let n = 0; ; n++) {
      const url = `https://site${n}-example.org/`; // its own registrable domain
      if (n > 10_000) throw new Error("no domain in that bucket range");
      if (want(bucketOfHash(domainHashOf(url)!))) return url;
    }
  };
  const trainUrl = domainWhere((b) => b < 80);
  const holdoutUrl = domainWhere((b) => b >= 90);

  test("train rows pass the train guard; holdout rows pass only the holdout one", () => {
    expect(
      routeRefusal(row({ source_url: trainUrl }), guard("train"), new Set(), new Set()),
    ).toBeNull();
    expect(
      routeRefusal(row({ source_url: holdoutUrl }), guard("train"), new Set(), new Set()),
    ).toBe("route_mismatch");
    expect(
      routeRefusal(row({ source_url: holdoutUrl }), guard("holdout"), new Set(), new Set()),
    ).toBeNull();
    expect(
      routeRefusal(row({ source_url: trainUrl }), guard("holdout"), new Set(), new Set()),
    ).toBe("route_mismatch");
  });

  test("a sealed domain is holdout; a gold domain never trains; a wrong hash is refused", () => {
    const sealedRow = row({ source_url: trainUrl });
    const hash = sealedRow.domain_hash;
    expect(routeRefusal(sealedRow, guard("train"), new Set([hash]), new Set())).toBe(
      "route_mismatch",
    );
    expect(routeRefusal(sealedRow, guard("holdout"), new Set([hash]), new Set())).toBeNull();
    expect(routeRefusal(sealedRow, guard("train", [hash]), new Set(), new Set([hash]))).toBe(
      "gold_domain",
    );
    expect(
      routeRefusal(
        { ...sealedRow, domain_hash: "0".repeat(16) },
        guard("train"),
        new Set(),
        new Set(),
      ),
    ).toBe("domain_hash_mismatch");
  });
});

describe("a16Row", () => {
  const base = {
    i: 0,
    href: "https://t.example.com/",
    matched: true,
    region: "content_block",
    regionSource: "model",
    regionConfidence: 0.912345,
    section: null,
    sectionSource: null,
    sectionConfidence: null,
    commentsRule: null,
    commentsRuleToken: null,
    rel: [],
  } as unknown as LinkPosition;

  test("the comments rule, then the section, then the region, then unknown", () => {
    expect(
      a16Row({
        ...base,
        commentsRule: "comments",
        commentsRuleToken: "comment-body",
        section: "comments",
        sectionSource: "rules",
        rel: ["ugc"],
      }),
    ).toEqual({
      i: 0,
      href: "https://t.example.com/",
      section: "comments",
      region: "content_block",
      rel: ["ugc"],
      conf: 1,
      evidence: "comment-body",
      source: "rule",
    });
    expect(
      a16Row({ ...base, section: "social_share", sectionSource: "model", sectionConfidence: 0.97 }),
    ).toMatchObject({
      section: "social_share",
      source: "model",
      conf: 0.97,
    });
    expect(a16Row(base)).toMatchObject({ section: "content_block", source: "model", conf: 0.9123 });
    expect(
      a16Row({ ...base, region: "footer", regionSource: "rules", regionConfidence: 1 }),
    ).toMatchObject({ section: "footer", source: "rule" });
    expect(
      a16Row({ ...base, region: "unknown", regionSource: null, regionConfidence: null }),
    ).toMatchObject({ section: "unknown", source: null, conf: null });
  });
});
