import { describe, expect, test } from "bun:test";
import {
  buildRequestV5,
  cleanText,
  COMPONENT_TYPES_V5,
  evaluateWithBackoff,
  HTML_PACKET_REVISION_V5,
  loadPolicy,
  looksLikeCode,
  nodeEvidence,
  nodeEvidenceFromHtml,
  PACKET_REVISION_V5,
  type ExtractedCandidate,
  PAGE_TYPES_V5,
  pageEvidence,
  PROMPT_REVISION_V5,
  PURPOSES_V5,
  REGIONS_V5,
  safePath,
  serializeNodePacket,
  toRowsV5,
} from "./jev-v5.ts";
import type { CapturedNode, CapturedPage } from "../labeler/types.ts";
import { extractPage } from "../extract-dom.ts";

function node(overrides: Partial<CapturedNode> & { id: string }): CapturedNode {
  return {
    parentId: null,
    tag: "div",
    role: null,
    text: "",
    selector: "html > body",
    rect: { x: 0, y: 0, width: 100, height: 50 },
    depth: 1,
    suggestion: null,
    ...overrides,
  };
}

function page(nodes: CapturedNode[], overrides: Partial<CapturedPage> = {}): CapturedPage {
  return {
    id: "page_test",
    url: "https://shop.example.com/collections/mugs?utm=x#frag",
    title: "Mugs",
    capturedAt: "2026-09-19T00:00:00.000Z",
    contentHash: "sha256:a",
    captureHash: "sha256:b",
    width: 1440,
    height: 3000,
    viewport: { width: 1440, height: 900, deviceScaleFactor: 1 },
    nodes,
    screenshotUrl: "/captures/page_test.png",
    split: "training-review",
    ...overrides,
  } as CapturedPage;
}

describe("cleanText", () => {
  test("drops a complete CSS rule but keeps the prose after it", () => {
    const value = cleanText("#a{position:fixed;top:0}@keyframes x{from{opacity:1}} Real prose.");
    expect(value).toContain("Real prose.");
    expect(value).not.toContain("position:fixed");
    expect(value).not.toContain("@keyframes");
  });

  test("redacts an email and a bearer token before anything else sees them", () => {
    const value = cleanText("write to sales@example.com with Bearer sk-abc123def");
    expect(value).toBe("write to [EMAIL] with [REDACTED]");
  });

  test("bounds the result", () => {
    expect(cleanText("x".repeat(900)).length).toBe(320);
  });
});

describe("looksLikeCode", () => {
  test.each([
    ['{"@context":"https://schema.org","@type":"WebSite","name":"N"}', true],
    ["(function(e){let t=1;return t})()", true],
    ["!function(){}()", true],
    ["window.dataLayer = window.dataLayer || []", true],
    ["Find your application identifier and rotate secret keys without downtime.", false],
    ["Pricing starts at $10 per month (billed annually).", false],
  ])("%s", (value, expected) => {
    expect(looksLikeCode(value as string)).toBe(expected);
  });
});

describe("packets from full HTML", () => {
  const html = `<!doctype html><html><body>
<header class="site-header"><a href="/"><img src="/logo.svg" alt="Northwind Coffee"></a>
<nav><ul><li><a href="/shop">Shop</a></li><li><a href="/about">About</a></li></ul></nav></header>
<main><figure><iframe src="https://www.youtube.com/embed/abc" title="How we roast"></iframe><figcaption>Inside the roastery</figcaption></figure></main>
</body></html>`;

  function candidateFor(tag: string) {
    const { candidates } = extractPage(html, "https://northwind.example/shop");
    return candidates.find((item) => (item as { tag: string }).tag === tag) as ExtractedCandidate;
  }

  test("an HTML packet declares its own revision and carries attributes", () => {
    const evidence = nodeEvidenceFromHtml(candidateFor("header"), "node_h");
    const packet = serializeNodePacket(evidence);
    expect(evidence.attributesAvailable).toBe(true);
    expect(packet.split("\n")[0]).toBe(HTML_PACKET_REVISION_V5);
    // The logo is the whole reason the media rule is answerable from HTML.
    expect(packet).toContain('media=img[alt="Northwind Coffee"|same-site]');
    expect(packet).toContain("classTokens=site-header");
  });

  test("a third-party embed arrives as a provider slug, never a hostname", () => {
    const packet = serializeNodePacket(nodeEvidenceFromHtml(candidateFor("figure"), "node_f"));
    expect(packet).toContain("|youtube]");
    expect(packet).not.toContain("youtube.com");
    expect(packet).not.toContain("northwind.example");
  });

  test("class tokens can be withheld, since the policy forbids labelling from them alone", () => {
    const evidence = nodeEvidenceFromHtml(candidateFor("header"), "node_h", {
      includeClassTokens: false,
    });
    expect(evidence.classTokens).toBeUndefined();
    expect(serializeNodePacket(evidence)).not.toContain("classTokens=");
  });

  test("a capture packet and an HTML packet are never confused for one another", () => {
    const captured = page([node({ id: "n1", tag: "header", text: "Shop About" })]);
    const fromCapture = serializeNodePacket(nodeEvidence(captured, captured.nodes[0]));
    const fromHtml = serializeNodePacket(nodeEvidenceFromHtml(candidateFor("header"), "node_h"));
    expect(fromCapture.split("\n")[0]).toBe(PACKET_REVISION_V5);
    expect(fromHtml.split("\n")[0]).toBe(HTML_PACKET_REVISION_V5);
    // HTML has no layout, so geometry is omitted rather than faked.
    expect(fromCapture).toContain("size=");
    expect(fromHtml).not.toContain("size=");
  });

  test("extractPage now reaches the nodes the rare component classes live on", () => {
    const { candidates } = extractPage(html, "https://northwind.example/shop");
    const tags = new Set(candidates.map((item) => (item as { tag: string }).tag));
    // Before the extractor was widened every one of these was absent, so no
    // sample size could yield an image, card, link or video_player label. The
    // per-parent cap and the container compatibility guarantee are pinned in
    // extract-dom.test.ts; this only asserts the packet builder can see them.
    for (const tag of ["img", "li", "a", "iframe"]) expect(tags.has(tag)).toBe(true);
    expect(tags.has("header")).toBe(true);
  });
});

describe("packets", () => {
  test("an element with no text still carries structure, not a bare token", () => {
    const image = node({
      id: "n2",
      tag: "img",
      parentId: "n1",
      rect: { x: 0, y: 0, width: 1440, height: 700 },
    });
    const captured = page([node({ id: "n1", tag: "section" }), image]);
    const packet = serializeNodePacket(nodeEvidence(captured, image));
    expect(packet).toContain("tag=img");
    expect(packet).toContain("size=1440x700");
    expect(packet).toContain("ancestors=section");
    expect(packet).toContain("text=[empty]");
  });

  test("subtree and children come from the captured node graph", () => {
    const footer = node({ id: "f", tag: "footer", parentId: null, text: "About us and contact" });
    const captured = page([
      footer,
      node({ id: "c1", tag: "nav", parentId: "f" }),
      node({ id: "c2", tag: "ul", parentId: "c1" }),
      node({ id: "c3", tag: "li", parentId: "c2" }),
      node({ id: "c4", tag: "li", parentId: "c2" }),
    ]);
    const packet = serializeNodePacket(nodeEvidence(captured, footer));
    expect(packet).toContain("children=nav");
    expect(packet).toContain("li*2");
  });

  test("page evidence prefers prose and drops script residue", () => {
    const captured = page([
      node({
        id: "s",
        tag: "script",
        text: '{"@context":"https://schema.org","@type":"X"}'.repeat(3),
      }),
      node({ id: "p", tag: "p", text: "A hand thrown stoneware mug fired at high temperature." }),
    ]);
    const evidence = pageEvidence(captured);
    expect(evidence.lines).toEqual(["A hand thrown stoneware mug fired at high temperature."]);
  });

  test("the page path is carried without host, query or fragment", () => {
    expect(safePath("https://shop.example.com/collections/mugs?utm=x#frag")).toBe(
      "/collections/mugs",
    );
  });
});

describe("buildRequestV5", () => {
  const policy = loadPolicy();
  const target = node({ id: "n1", tag: "footer", text: "Contact and legal" });
  const captured = page([target]);

  test("asks one page Choice over eleven classes, never sixty-five Nouls", () => {
    const { request } = buildRequestV5(captured, [target], policy);
    expect(request.questions.page_type?.type).toBe("choice");
    expect(Object.keys(request.questions.page_type!.criteria!)).toEqual([...PAGE_TYPES_V5]);
    expect(PAGE_TYPES_V5.length).toBe(11);
    expect(Object.keys(request.questions).filter((key) => key.startsWith("page_type_"))).toEqual(
      [],
    );
  });

  test("per node: one region Noul each, one purpose Noul each, one component Choice", () => {
    const { request } = buildRequestV5(captured, [target], policy);
    const keys = Object.keys(request.questions);
    expect(keys.filter((key) => key.includes("_region_")).length).toBe(REGIONS_V5.length);
    expect(keys.filter((key) => key.includes("_purpose_")).length).toBe(PURPOSES_V5.length);
    expect(keys.filter((key) => key.endsWith("_component_type")).length).toBe(1);
    expect(keys.length).toBe(1 + REGIONS_V5.length + PURPOSES_V5.length + 1);
  });

  test("the reduced purpose vocabulary is twelve and excludes the unsupported values", () => {
    expect(PURPOSES_V5.length).toBe(12);
    expect(PURPOSES_V5).toContain("navigation");
    expect(PURPOSES_V5).not.toContain("media_playback");
    expect(PURPOSES_V5).not.toContain("unknown");
  });

  test("the component vocabulary is unchanged, so v4 and v5 stay comparable", () => {
    const { request } = buildRequestV5(captured, [target], policy);
    const criteria = request.questions[`node_n1_component_type`]!.criteria!;
    expect(Object.keys(criteria).sort()).toEqual([...COMPONENT_TYPES_V5].sort());
  });

  test("state carries the policy bytes and its hash", () => {
    const { request } = buildRequestV5(captured, [target], policy);
    const state = request.state as unknown as { policy: { text: string; sha256: string } };
    expect(state.policy.text).toBe(policy.text);
    expect(state.policy.sha256).toBe(policy.sha256);
    expect(state.policy.text).toContain("label the site-wide chrome `layout_container`");
  });

  test("no hostname reaches the request", () => {
    const { request } = buildRequestV5(captured, [target], policy);
    expect(JSON.stringify(request)).not.toContain("shop.example.com");
  });

  test("omitting the page question drops it without touching node questions", () => {
    const { request } = buildRequestV5(captured, [target], policy, { includePage: false });
    expect(request.questions.page_type).toBeUndefined();
    expect(Object.keys(request.questions).length).toBe(REGIONS_V5.length + PURPOSES_V5.length + 1);
  });
});

describe("toRowsV5", () => {
  const policy = loadPolicy();
  const target = node({ id: "n1", tag: "footer", text: "Contact and legal" });
  const captured = page([target]);

  test("maps a Choice to one label and thresholds Nouls at 0.75", () => {
    const built = buildRequestV5(captured, [target], policy);
    const answers: Record<string, unknown> = {
      page_type: { type: "choice", choice: "product_listing", probabilities: {}, confidence: 0.9 },
      node_n1_component_type: { type: "choice", choice: "layout_container", probabilities: {} },
    };
    for (const label of REGIONS_V5)
      answers[`node_n1_region_${label}`] = { type: "noul", noul: label === "footer" ? 0.9 : 0.74 };
    for (const label of PURPOSES_V5)
      answers[`node_n1_purpose_${label}`] = {
        type: "noul",
        noul: label === "navigation" ? 0.75 : 0.2,
      };
    const rows = toRowsV5(
      captured,
      built,
      { model: "jev-1.13.0", answers: answers as never },
      {
        split: "validation",
        strata: new Map([["n1", "footer"]]),
        policy,
        latencyMs: 5,
      },
      true,
    );
    const pageRow = rows.find((row) => row.recordType === "page")!;
    const nodeRow = rows.find((row) => row.recordType === "node")!;
    expect(pageRow.observedAxes.pageType).toEqual({ complete: true, labels: "product_listing" });
    expect(nodeRow.observedAxes.componentType).toEqual({
      complete: true,
      labels: "layout_container",
    });
    // 0.74 is below the threshold and 0.75 is exactly at it.
    expect(nodeRow.observedAxes.regions).toEqual({ complete: true, labels: ["footer"] });
    expect(nodeRow.observedAxes.purposes).toEqual({ complete: true, labels: ["navigation"] });
    expect(nodeRow.stratum).toBe("footer");
    expect(nodeRow.promptRevision).toBe(PROMPT_REVISION_V5);
    expect(nodeRow.promptRevision).toBe(policy.revision);
    expect(nodeRow.policySha256).toBe(policy.sha256);
  });
});

describe("policy revisions", () => {
  test("each revision loads its own file and declares its own prompt revision", () => {
    expect(loadPolicy("v5").revision).toBe("dom-suggestions-v5");
    expect(loadPolicy("v6").revision).toBe("dom-suggestions-v6");
    expect(loadPolicy("v5").sha256).not.toBe(loadPolicy("v6").sha256);
  });

  test("v6 deletes the link-menu escape hatch that decided 46% of the v5 gap", () => {
    const v5 = loadPolicy("v5").text;
    const v6 = loadPolicy("v6").text;
    expect(v5).toContain("dominated by a link menu, `navigation_menu` wins");
    expect(v6).not.toContain("dominated by a link menu, `navigation_menu` wins");
    expect(v6).toContain("are `layout_container`. No exception.");
  });

  test("v6 gates the no-attributes abstention on the evidence being available", () => {
    const v6 = loadPolicy("v6").text;
    expect(v6).toContain("When attributes are unavailable, label from the tag");
    expect(v6).toContain("img, no attributes available (capture packet)         -> image");
  });

  test("v6 defines the pairs v5 left undefined", () => {
    const v6 = loadPolicy("v6").text;
    expect(v6).toContain("`image` versus `media`");
    expect(v6).toContain("the item is a `card`, the wrapper is a `list`");
  });

  test("v6 deliberately leaves the <main> rule alone", () => {
    const rule =
      "label it\n`layout_container` when it only wraps the page's sections, and\n`content_section` when it is itself the single body of content";
    expect(loadPolicy("v6").text).toContain(rule);
    expect(loadPolicy("v5").text).toContain(rule);
  });

  test("an unknown revision is refused rather than silently defaulted", () => {
    expect(() => loadPolicy("v99" as never)).toThrow("unknown policy revision");
  });

  test("v7 is the frozen revision: v6 plus the shape-only scoping", () => {
    const v7 = loadPolicy("v7").text;
    expect(v7).toContain("This rule decides SHAPE only");
    expect(v7).toContain("never suppresses a purpose");
    // Everything v6 fixed is still in v7.
    expect(v7).toContain("are `layout_container`. No exception.");
    expect(v7).toContain("When attributes are unavailable, label from the tag");
  });
});

describe("evaluateWithBackoff", () => {
  test("retries a 429 and reports the retry count", async () => {
    let calls = 0;
    const result = await evaluateWithBackoff(
      async () => {
        calls++;
        if (calls < 2) throw new Error("TypeSafe HTTP 429 (rate_limited)");
        return { model: "jev-1.13.0", answers: {} };
      },
      { model: "jev-1.13.0", state: {} as never, questions: {} },
      3,
    );
    expect(calls).toBe(2);
    expect(result.retries).toBe(1);
  });

  test("does not retry a 422, which never becomes valid by waiting", async () => {
    let calls = 0;
    await expect(
      evaluateWithBackoff(
        async () => {
          calls++;
          throw new Error("TypeSafe HTTP 422 (invalid_request)");
        },
        { model: "jev-1.13.0", state: {} as never, questions: {} },
        3,
      ),
    ).rejects.toThrow("422");
    expect(calls).toBe(1);
  });
});
