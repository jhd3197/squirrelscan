// The deterministic comments rule (#2307): which markup is a comment section,
// which look-alikes are not, and the wrapper and reviews cases.

import { describe, expect, test } from "bun:test";

import { parseHTML } from "../../../../packages/parser/src/dom.ts";
import {
  commentSectionOf,
  findCommentSections,
  hasUgcRel,
  isCommentToken,
} from "./comments-rule.ts";

const page = (inner: string) =>
  parseHTML(
    `<html><body><h1>Post title</h1><article><p>Body text</p></article>${inner}</body></html>`,
  ).document;
const linkIn = (inner: string, selector = "a.target") => {
  const el = page(inner).querySelector(selector);
  if (!el) throw new Error(`no ${selector}`);
  return commentSectionOf(el);
};

describe("isCommentToken", () => {
  test.each([
    ["comments", "id"],
    ["comment-12", "id"],
    ["li-comment-12", "id"],
    ["div-comment-12", "id"],
    ["respond", "id"],
    ["disqus_thread", "id"],
    ["commentlist", "class"],
    ["comment-list", "class"],
    ["comments-area", "class"],
    ["comment-body", "class"],
    ["comment", "class"],
    ["ast-comment-list", "class"],
    ["wp-block-post-comments", "class"],
    ["commentsBox", "id"],
    ["comment_text", "class"],
    ["comment-respond", "class"],
  ] as const)("%s (%s) is a comment section token", (token, kind) => {
    expect(isCommentToken(token, kind)).toBe(true);
  });

  test.each([
    ["comments-link", "class"],
    ["comment-count", "class"],
    ["comments-number", "class"],
    ["post-meta-comments", "class"],
    ["recent-comments", "class"],
    ["widget_recent_comments", "class"],
    ["recentcomments", "id"],
    ["has-comments", "class"],
    ["comments-closed", "class"],
    ["no-comments", "class"],
    ["nocomments", "class"],
    ["commentary", "class"],
    ["respond", "class"],
    ["comment-reply-link", "class"],
  ] as const)("%s (%s) is not", (token, kind) => {
    expect(isCommentToken(token, kind)).toBe(false);
  });
});

describe("commentSectionOf", () => {
  test("a link in a WordPress comment author line is in the comments", () => {
    const match = linkIn(
      `<div id="comments" class="comments-area"><ol class="comment-list"><li id="comment-7" class="comment">
        <article id="div-comment-7" class="comment-body"><footer class="comment-meta"><div class="comment-author">
        <a class="url target" rel="external nofollow ugc" href="https://spam.example/">Cheap pills</a></div></footer>
        <div class="comment-content"><p>Great post</p></div></article></li></ol></div>`,
    );
    expect(match?.label).toBe("comments");
    expect(match?.container.getAttribute("class")).toBe("comment-author");
    expect(match?.token).toBe("comment-author");
  });

  test("a link in the reply form is in the comments", () => {
    const match = linkIn(
      `<div id="respond"><form id="commentform"><p class="logged-in-as"><a class="target" href="/wp-login.php">Log in</a></p></form></div>`,
    );
    expect(match?.label).toBe("comments");
  });

  test("the article's own links and a comment-count link are not", () => {
    expect(linkIn(`<p><a class="target" href="https://a.example/">source</a></p>`)).toBeNull();
    expect(
      linkIn(
        `<div class="entry-meta"><span class="comments-link"><a class="target" href="#comments">3 comments</a></span></div>`,
      ),
    ).toBeNull();
  });

  test("a recent-comments widget in the sidebar is not", () => {
    expect(
      linkIn(
        `<aside><section class="widget widget_recent_comments"><ul id="recentcomments"><li class="recentcomments">
          <a class="target" href="https://x.example/">Ann</a></li></ul></section></aside>`,
      ),
    ).toBeNull();
  });

  test("a wrapper holding the page title is refused, and nothing above it counts", () => {
    const doc = parseHTML(
      `<html><body><div class="site comments"><h1>Title</h1><p><a class="target" href="https://a.example/">x</a></p></div></body></html>`,
    ).document;
    expect(commentSectionOf(doc.querySelector("a.target")!)).toBeNull();
  });

  test("WooCommerce product reviews are reviews", () => {
    const match = linkIn(
      `<div id="reviews" class="woocommerce-Reviews"><div id="comments"><ol class="commentlist"><li class="review">
        <a class="target" href="https://b.example/">Bob</a></li></ol></div></div>`,
    );
    expect(match?.label).toBe("reviews");
  });

  test("hops count from the link up to the matching element", () => {
    const match = linkIn(
      `<div id="comments"><p><span><a class="target" href="x">x</a></span></p></div>`,
    );
    expect(match?.hops).toBe(3);
    expect(match?.token).toBe("comments");
  });
});

describe("hasUgcRel and findCommentSections", () => {
  test("rel=ugc is read as a token", () => {
    const doc = page(
      `<a id="u" rel="nofollow UGC" href="x">x</a><a id="n" rel="nofollow" href="y">y</a>`,
    );
    expect(hasUgcRel(doc.querySelector("#u")!)).toBe(true);
    expect(hasUgcRel(doc.querySelector("#n")!)).toBe(false);
  });

  test("one outermost section per thread, none on a page without comments", () => {
    const found = findCommentSections(
      page(
        `<div id="comments"><ol class="commentlist"><li class="comment">a</li><li class="comment">b</li></ol></div>`,
      ),
    );
    expect(found.map((el) => el.getAttribute("id"))).toEqual(["comments"]);
    expect(findCommentSections(page(`<footer><a href="x">x</a></footer>`))).toEqual([]);
  });
});
