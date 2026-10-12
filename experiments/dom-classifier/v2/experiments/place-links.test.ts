// The place-links CLI step (#2307): JSON line in, JSON line out, errors per line.

import { describe, expect, test } from "bun:test";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { gzipSync } from "node:zlib";

import { placeRequest } from "./place-links.ts";

const HTML = `<!doctype html><html><head><title>t</title></head><body>
<header><nav><a href="https://own.example.org/about">About</a></nav></header>
<main><article><h1>Post</h1><p>See <a href="https://src.example.net/r">the source</a>.</p></article>
<div id="comments" class="comments-area"><ol class="comment-list"><li class="comment">
<div class="comment-author"><a rel="nofollow ugc" href="https://spam.example.com/">Pills</a></div></li></ol></div></main>
<footer><a href="https://shop.example.com/">Shop</a></footer></body></html>`;

const run = (request: object) => placeRequest(JSON.stringify(request), null, 500) as any;

describe("placeRequest", () => {
  test("places listed links: the comment link is comments, the body link is not", () => {
    const out = run({
      id: "p1",
      url: "https://own.example.org/post",
      html: HTML,
      links: [{ href: "https://spam.example.com/" }, { href: "https://src.example.net/r" }],
    });
    expect(out.ok).toBe(true);
    expect(out.id).toBe("p1");
    expect(out.degraded).toBe("no model artifact");
    expect(out.comment_sections).toEqual([{ label: "comments", evidence: "comments" }]);
    const [spam, source] = out.links;
    expect(spam).toMatchObject({
      section: "comments",
      source: "rule",
      conf: 1,
      evidence: "comment-author",
      rel: ["ugc", "nofollow"],
      located: true,
      in_walk: true,
    });
    expect(source.section).not.toBe("comments");
  });

  test("no links given: every outbound link, internal ones left out", () => {
    const out = run({ id: "p2", url: "https://own.example.org/post", html: HTML });
    expect(out.links.map((l: any) => l.href).sort()).toEqual([
      "https://shop.example.com/",
      "https://spam.example.com/",
      "https://src.example.net/r",
    ]);
  });

  test("reads html_path, plain or gzipped", () => {
    const dir = mkdtempSync(join(tmpdir(), "place-"));
    writeFileSync(join(dir, "a.html"), HTML);
    writeFileSync(join(dir, "b.html.gz"), gzipSync(HTML));
    for (const name of ["a.html", "b.html.gz"]) {
      const out = run({
        id: name,
        url: "https://own.example.org/post",
        html_path: join(dir, name),
      });
      expect(out.ok).toBe(true);
      expect(out.links.length).toBe(3);
    }
  });

  test("a bad line is an error row, never a throw", () => {
    expect(run({ id: "x", url: "https://a.example/" })).toEqual({
      id: "x",
      ok: false,
      error: "give html or html_path, not both and not neither",
    });
    expect(run({ id: "y", html: "<p>" }).error).toBe("url is required");
    expect((placeRequest("not json", null, 500) as any).ok).toBe(false);
    expect(run({ id: "z", url: "https://a.example/", html_path: "/nonexistent/x.html" }).ok).toBe(
      false,
    );
  });

  test("a link not on the page is reported, not dropped", () => {
    const out = run({
      id: "p3",
      url: "https://own.example.org/post",
      html: HTML,
      links: [{ href: "https://nowhere.example.org/" }],
    });
    expect(out.links[0]).toMatchObject({ located: false, section: "unknown", region: "unknown" });
  });
});
