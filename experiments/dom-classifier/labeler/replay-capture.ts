import { lookup } from "node:dns/promises";
import type { Browser, Frame, LaunchOptions, Route } from "playwright";
import { browserLaunchOptions } from "./browser-runtime.ts";
import { isPublicAddress, startEgressGuard, type EgressGuard } from "./egress-guard.ts";
import {
  capturePixels,
  clipNodes,
  digest,
  isPrivateHostname,
  MAX_CAPTURE_HEIGHT,
  MAX_LAZY_LOAD_STEPS,
  nodeId,
  pngDimensions,
  VIEWPORT,
  weakSuggestion,
  type RawNode,
} from "./capture.ts";
import { LabelStore } from "./store.ts";
import type { CapturedNode, CapturedPage } from "./types.ts";

/** The attribute a caller puts on stored elements it needs located in the render. */
export const REPLAY_MARKER = "data-replay-target";
const MAX_REPLAY_MS = 45_000;
/**
 * Requests that would refetch content or report the visit; stored bytes are the
 * content. WebSockets never reach `route` at all, so they are closed by a
 * separate WebSocket route.
 */
const BLOCKED_RESOURCE_TYPES = new Set(["xhr", "fetch", "eventsource", "ping"]);

/**
 * Replayed pages run third-party scripts on a developer machine where local
 * services listen. All their traffic goes through the egress guard (see
 * `egress-guard.ts`), and WebRTC may not open non-proxied UDP, so it can
 * neither leak local addresses nor reach the LAN around the guard.
 *
 * Chromium's Private/Local Network Access features were tried as a second
 * guard and measured ineffective in the bundled headless shell (140): a public
 * page still reached 127.0.0.1 with them on. They are deliberately not set.
 */
export const REPLAY_BROWSER_ARGS = [
  "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
] as const;

/** Launch options for the browser a caller passes to `captureReplayTarget`. */
export function replayLaunchOptions(): LaunchOptions {
  return { ...browserLaunchOptions(), args: [...REPLAY_BROWSER_ARGS] };
}

/**
 * A historical replay renders STORED html instead of fetching the page, so a
 * reviewer judges the bytes a model was given. The navigation to `url` is
 * answered with `html`; stylesheets, images, fonts and scripts still load from
 * their public hosts (`assetMode: "hybrid"`), while data requests, beacons and
 * any further top-level navigation are blocked.
 */
export type ReplayTarget = {
  html: string;
  url: string;
  originalCrawledAt?: string | null;
  /** Private opaque corpus reference; never a path. */
  corpusRef?: string | null;
  /** Values of `REPLAY_MARKER` attributes to locate in the render. */
  markers?: string[];
};

export type ReplayLocation = {
  nodeId: string | null;
  /** False when the element rendered hidden, empty, or outside the bounded screenshot. */
  visible: boolean;
  /** False when the marked element is gone from the rendered DOM (a script replaced it). */
  present: boolean;
};

export type ReplayResult = {
  page: CapturedPage;
  located: Record<string, ReplayLocation>;
  /** Cheap render-quality signals for an operator to hold out blank or broken replays. */
  quality: { visibleNodes: number; visibleTextChars: number; documentHeight: number };
};

/** Scripts are off unless the caller turns them on for a page that needs them. */
type ReplayOptions = {
  scripts?: boolean;
  timeoutMs?: number;
  maxHeight?: number;
  /** Defaults to one guard shared by every replay in this process. */
  egress?: EgressGuard;
};

let sharedGuard: Promise<EgressGuard> | null = null;
/** The process-wide egress guard, started on first use. */
export function replayEgressGuard() {
  sharedGuard ??= startEgressGuard();
  return sharedGuard;
}

/**
 * An early, cheap refusal of hosts that resolve private, so a blocked asset
 * fails at the route instead of opening a proxy tunnel. It is not the guard
 * against DNS rebinding: this lookup and the connection are separate. The
 * egress guard resolves, checks and connects to one address, and that is what
 * enforces the rule.
 */
const publicHosts = new Map<string, Promise<boolean>>();
const DNS_TIMEOUT_MS = 3_000;
function isPublicHost(hostname: string) {
  if (isPrivateHostname(hostname)) return Promise.resolve(false);
  if (!publicHosts.has(hostname)) {
    // A resolver that stalls must not hold a route open: a slow answer is a block.
    const timeout = new Promise<boolean>((resolve) => setTimeout(resolve, DNS_TIMEOUT_MS, false));
    const resolved = lookup(hostname, { all: true, verbatim: true }).then(
      (addresses) =>
        addresses.length > 0 && addresses.every((entry) => isPublicAddress(entry.address)),
      () => false,
    );
    publicHosts.set(hostname, Promise.race([resolved, timeout]));
  }
  return publicHosts.get(hostname)!;
}

function replayUrl(raw: string) {
  const url = new URL(raw);
  if (!/^https?:$/.test(url.protocol) || !url.hostname || url.username || url.password)
    throw new Error("Replay URL must be a public http(s) URL");
  if (isPrivateHostname(url.hostname)) throw new Error("Private network URL is not allowed");
  url.hash = "";
  return url;
}

/**
 * Render one stored page in an isolated context of a caller-owned browser and
 * write it to `store` as an immutable `historical_replay` capture.
 */
export async function captureReplayTarget(
  store: LabelStore,
  browser: Browser,
  target: ReplayTarget,
  options: ReplayOptions = {},
): Promise<ReplayResult> {
  const documentUrl = replayUrl(target.url);
  const scripts = options.scripts ?? false;
  const maxHeight = Math.min(
    Math.max(1, options.maxHeight ?? MAX_CAPTURE_HEIGHT),
    MAX_CAPTURE_HEIGHT,
  );
  const timeoutMs = Math.min(Math.max(5_000, options.timeoutMs ?? MAX_REPLAY_MS), MAX_REPLAY_MS);
  const deadline = Date.now() + timeoutMs;
  const remaining = () => {
    const milliseconds = deadline - Date.now();
    if (milliseconds <= 0) throw new Error("Replay exceeded its deadline");
    return milliseconds;
  };
  const egress = options.egress ?? (await replayEgressGuard());
  const context = await browser.newContext({
    viewport: { width: VIEWPORT.width, height: VIEWPORT.height },
    deviceScaleFactor: VIEWPORT.deviceScaleFactor,
    serviceWorkers: "block",
    // Every request the page makes leaves through the guard, loopback included.
    proxy: { server: egress.server },
    userAgent: "SquirrelScan DOM labeling replay/1.0",
  });
  try {
    // WebSockets bypass `route`, so every socket is answered here and closed
    // before it reaches a server.
    await context.routeWebSocket(/.*/, (socket) => socket.close());
    const page = await context.newPage();
    // A popup is a second document this replay did not ask for.
    context.on("page", (opened) => {
      if (opened !== page) void opened.close().catch(() => undefined);
    });
    let served = false;
    // Context-level, so requests from popups and workers pass the same checks.
    await context.route("**/*", async (route: Route) => {
      const request = route.request();
      const url = new URL(request.url());
      // Only a navigation is asked for its frame (worker requests have none). A
      // popup's first navigation has no frame yet, and a popup is not ours.
      let frame: Frame | null = null;
      if (request.isNavigationRequest()) {
        try {
          frame = request.frame();
        } catch {
          return route.abort("blockedbyclient");
        }
        if (frame.page() !== page) return route.abort("blockedbyclient");
      }
      if (frame === page.mainFrame()) {
        // The first navigation is the stored document; anything after it
        // (a script or meta redirect) would leave the stored bytes, and a 204
        // keeps the current document where an abort would show an error page.
        if (!served && url.href.split("#")[0] === documentUrl.href) {
          served = true;
          return route.fulfill({
            status: 200,
            contentType: "text/html; charset=utf-8",
            // Scripts off is a policy on the document, not on the context:
            // the settle and node walk below still need to evaluate.
            headers: scripts ? {} : { "content-security-policy": "script-src 'none'" },
            body: target.html,
          });
        }
        return route.fulfill({ status: 204 });
      }
      if (
        BLOCKED_RESOURCE_TYPES.has(request.resourceType()) ||
        !/^https?:$/.test(url.protocol) ||
        !(await isPublicHost(url.hostname))
      )
        return route.abort("blockedbyclient");
      return route.continue();
    });
    page.setDefaultTimeout(Math.min(20_000, remaining()));
    await page.goto(documentUrl.href, {
      waitUntil: "domcontentloaded",
      timeout: Math.min(20_000, remaining()),
    });
    let renderSettled: CapturedPage["renderSettled"] = "networkidle";
    try {
      await page.waitForLoadState("networkidle", { timeout: Math.min(6_000, remaining()) });
    } catch {
      renderSettled = "bounded_timeout";
    }
    await page.evaluate(settleForReplay, MAX_LAZY_LOAD_STEPS);
    remaining();
    const result = await page.evaluate(extractReplayNodes, REPLAY_MARKER);
    const indexed = result.nodes;
    const nodes: CapturedNode[] = indexed.map((node) => ({
      id: nodeId(node.selector),
      parentId: node.parentIndex === null ? null : nodeId(indexed[node.parentIndex]!.selector),
      tag: node.tag,
      role: node.role,
      text: node.text,
      selector: node.selector,
      rect: node.rect,
      depth: node.depth,
      suggestion: weakSuggestion(node),
    }));
    const screenshot = await capturePixels(page, result.documentHeight, maxHeight, remaining());
    const dimensions = pngDimensions(screenshot);
    const boundedNodes = clipNodes(nodes, dimensions.width, dimensions.height);
    const kept = new Map(boundedNodes.map((node) => [node.id, node]));
    const located: Record<string, ReplayLocation> = {};
    for (const marker of target.markers ?? []) {
      const index = result.markers[marker];
      const node = index === undefined ? undefined : nodes[index];
      const bounded = node ? kept.get(node.id) : undefined;
      located[marker] = {
        nodeId: node?.id ?? null,
        // A target must be on the screenshot with most of its box, or the
        // outline a reviewer sees is not the element that was labelled.
        visible: Boolean(
          node &&
          bounded &&
          bounded.rect.width * bounded.rect.height >= 0.5 * node.rect.width * node.rect.height,
        ),
        present: index !== undefined || result.hiddenMarkers.includes(marker),
      };
    }
    const contentHash = `sha256:${digest(result.html)}`;
    const captureHash = `sha256:${digest(`${contentHash}\n${digest(screenshot)}\n${VIEWPORT.width}x${VIEWPORT.height}@${VIEWPORT.deviceScaleFactor}`)}`;
    const pageId = `page_${captureHash.slice("sha256:".length, 24 + "sha256:".length)}`;
    const captured: CapturedPage = {
      id: pageId,
      url: documentUrl.href,
      title: result.title || documentUrl.hostname,
      capturedAt: new Date().toISOString(),
      contentHash,
      captureHash,
      width: dimensions.width,
      height: dimensions.height,
      viewport: VIEWPORT,
      nodes: boundedNodes,
      screenshotUrl: `/captures/${pageId}.png`,
      split: "training-review",
      documentHeight: Math.ceil(result.documentHeight),
      heightCapped: result.documentHeight > dimensions.height,
      renderSettled,
      sourceKind: "historical_replay",
      originalCrawledAt: target.originalCrawledAt ?? null,
      sourceUrl: documentUrl.href,
      corpusRef: target.corpusRef ?? null,
      assetMode: "hybrid",
      scriptMode: scripts ? "enabled" : "disabled",
    };
    store.writeCapture(captured, screenshot);
    return {
      page: captured,
      located,
      quality: {
        visibleNodes: boundedNodes.length,
        visibleTextChars: result.visibleTextChars,
        documentHeight: Math.ceil(result.documentHeight),
      },
    };
  } finally {
    await context.close().catch(() => undefined);
  }
}

/**
 * Same settle as a fresh capture (no animation, lazy-load scroll, back to the
 * origin), plus one replay rule: fixed elements become absolute at their
 * current place, so a full-page screenshot draws them where they were measured.
 */
async function settleForReplay(lazyLoadSteps: number) {
  const style = document.createElement("style");
  style.textContent =
    "html,body{scroll-behavior:auto!important;scroll-snap-type:none!important}*,*::before,*::after{animation:none!important;transition:none!important;caret-color:transparent!important}";
  document.head.append(style);
  await Promise.race([
    document.fonts?.ready ?? Promise.resolve(),
    new Promise((resolve) => setTimeout(resolve, 2_000)),
  ]);
  const step = Math.max(450, Math.floor(window.innerHeight * 0.8));
  let y = 0;
  for (let index = 0; index < lazyLoadSteps; index += 1) {
    const maximumY = Math.max(0, document.documentElement.scrollHeight - window.innerHeight);
    window.scrollTo(0, Math.min(y, maximumY));
    await new Promise((resolve) => setTimeout(resolve, 25));
    if (y >= maximumY) break;
    y += step;
  }
  window.scrollTo({ left: 0, top: 0, behavior: "auto" });
  document.documentElement.scrollTop = 0;
  if (document.body) document.body.scrollTop = 0;
  for (let frame = 0; frame < 8 && (window.scrollX !== 0 || window.scrollY !== 0); frame += 1) {
    await new Promise((resolve) => requestAnimationFrame(resolve));
    window.scrollTo({ left: 0, top: 0, behavior: "auto" });
  }
  if (window.scrollX !== 0 || window.scrollY !== 0)
    throw new Error("Could not settle replay at page origin");
  // Measure every fixed element first (at the origin, viewport and document
  // coordinates agree), then pin each one: an absolute box is placed against its
  // containing block's padding box, which may be a positioned ancestor.
  const fixed = Array.from(document.querySelectorAll("body *"))
    .filter((element) => getComputedStyle(element).position === "fixed")
    .map((element) => ({ element: element as HTMLElement, rect: element.getBoundingClientRect() }));
  for (const { element, rect } of fixed) {
    const inline = element.style;
    inline.setProperty("position", "absolute", "important");
    inline.setProperty("box-sizing", "border-box", "important");
    inline.setProperty("width", `${rect.width}px`, "important");
    inline.setProperty("height", `${rect.height}px`, "important");
    inline.setProperty("bottom", "auto", "important");
    inline.setProperty("right", "auto", "important");
    inline.setProperty("margin", "0", "important");
    // A static <body> is reported as the offset parent but is not the containing block.
    const parent = element.offsetParent as HTMLElement | null;
    const container =
      parent === document.body && getComputedStyle(parent).position === "static" ? null : parent;
    const base = container
      ? container.getBoundingClientRect()
      : ({ top: -window.scrollY, left: -window.scrollX } as DOMRect);
    const top = rect.top - base.top - (container?.clientTop ?? 0) + (container?.scrollTop ?? 0);
    const left =
      rect.left - base.left - (container?.clientLeft ?? 0) + (container?.scrollLeft ?? 0);
    inline.setProperty("top", `${top}px`, "important");
    inline.setProperty("left", `${left}px`, "important");
  }
}

/**
 * The fresh-capture node walk (same filters and selector, so node ids agree),
 * except that a marked element is never dropped by the 5,000-node cap, and a
 * node whose parent was filtered out hangs from its nearest kept ancestor.
 */
function extractReplayNodes(marker: string) {
  const cleanText = (value: string) => value.replace(/\s+/g, " ").trim().slice(0, 280);
  const cleanRole = (value: string | null) =>
    value && /^[a-z][a-z-]{0,78}$/i.test(value) ? value.toLowerCase() : null;
  const selectorFor = (element: Element) => {
    const pieces: string[] = [];
    for (
      let current: Element | null = element;
      current && current.nodeType === Node.ELEMENT_NODE;
      current = current.parentElement
    ) {
      const tag = current.tagName.toLowerCase();
      if (tag === "html") {
        pieces.unshift("html");
        break;
      }
      const peers = current.parentElement
        ? Array.from(current.parentElement.children).filter(
            (item) => item.tagName === current!.tagName,
          )
        : [current];
      pieces.unshift(`${tag}:nth-of-type(${peers.indexOf(current) + 1})`);
    }
    return pieces.join(" > ");
  };
  const raw: Array<RawNode & { parentIndex: number | null }> = [];
  const byElement = new Map<Element, number>();
  const markers: Record<string, number> = {};
  const hiddenMarkers: string[] = [];
  let visibleTextChars = 0;
  for (const element of Array.from(document.querySelectorAll("body, body *"))) {
    const markerValue = element.getAttribute(marker);
    if (raw.length >= 5_000 && markerValue === null) continue;
    const tag = element.tagName.toLowerCase();
    if (["script", "style", "noscript", "template", "svg", "path", "meta", "link"].includes(tag)) {
      if (markerValue !== null) hiddenMarkers.push(markerValue);
      continue;
    }
    const computed = getComputedStyle(element);
    const rect = element.getBoundingClientRect();
    const hidden =
      computed.display === "none" ||
      computed.visibility === "hidden" ||
      Number(computed.opacity) === 0 ||
      rect.width < 2 ||
      rect.height < 2;
    if (hidden) {
      if (markerValue !== null) hiddenMarkers.push(markerValue);
      continue;
    }
    let depth = 0;
    for (let parent = element.parentElement; parent; parent = parent.parentElement) depth += 1;
    const selector = selectorFor(element);
    if (selector.length > 1_000 || depth > 160) {
      if (markerValue !== null) hiddenMarkers.push(markerValue);
      continue;
    }
    let parentIndex: number | null = null;
    for (
      let parent = element.parentElement;
      parent && parentIndex === null;
      parent = parent.parentElement
    )
      parentIndex = byElement.get(parent) ?? null;
    for (const child of Array.from(element.childNodes))
      if (child.nodeType === Node.TEXT_NODE)
        visibleTextChars += child.textContent?.trim().length ?? 0;
    raw.push({
      tag: tag.slice(0, 40),
      role: cleanRole(element.getAttribute("role")),
      text: cleanText(element.textContent ?? ""),
      selector,
      rect: {
        x: Math.round(rect.x + window.scrollX),
        y: Math.round(rect.y + window.scrollY),
        width: Math.round(rect.width),
        height: Math.round(rect.height),
      },
      depth,
      parentIndex,
    });
    byElement.set(element, raw.length - 1);
    if (markerValue !== null) markers[markerValue] = raw.length - 1;
  }
  const root = document.documentElement;
  return {
    title: cleanText(document.title),
    html: root.outerHTML,
    documentHeight: Math.max(root.scrollHeight, document.body?.scrollHeight ?? 0),
    nodes: raw,
    markers,
    hiddenMarkers,
    visibleTextChars,
  };
}
