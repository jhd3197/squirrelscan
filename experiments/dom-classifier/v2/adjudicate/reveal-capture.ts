/**
 * Forced-reveal screenshots of section targets that a replay leaves hidden (#2307).
 *
 * A third of section targets do not display when stored HTML is replayed: menus,
 * dropdowns, dialogs, consent banners and locale pickers are hidden until a
 * script, a click or a hover shows them. This loads the stored document under
 * the labeller's replay rules, then forces ONE marked target and its ancestor
 * chain visible (display, visibility, opacity, collapsed height and overflow,
 * clip, transforms, off-screen positioning, `hidden`, closed `<details>` and
 * `<dialog>`) and takes an element screenshot of the target alone.
 *
 * Network rules are `labeler/replay-capture.ts`'s, copied because that file
 * keeps them internal: the stored document answers the first navigation; any
 * later navigation is a 204; data requests, beacons, websockets, popups,
 * service workers and non-public hosts are refused; every request leaves
 * through the shared egress guard. `allowData` lifts ONLY the data-request block
 * (fetch/XHR/EventSource), so consent managers can load their configuration; it
 * is for sandboxed Modal runs and never for this machine.
 */
import { lookup } from "node:dns/promises";
import type { Browser, Frame, Route } from "playwright";
import { isPrivateHostname, isPublicAddress, VIEWPORT } from "../../labeler/capture.ts";
import { replayEgressGuard, REPLAY_MARKER } from "../../labeler/replay-capture.ts";
import { promoteDeferredStyles } from "./replay-html.ts";

const DATA_TYPES = new Set(["xhr", "fetch", "eventsource"]);
const ALWAYS_BLOCKED = new Set(["ping"]);
const PAGE_DEADLINE_MS = 40_000;
const PAD = 24;
const MAX_CLIP_HEIGHT = 2_000;

export type RevealRect = { x: number; y: number; width: number; height: number };
export type RevealResult = {
  marker: string;
  found: boolean;
  /** The target has a box of at least 8 x 8 after the overrides. */
  revealed: boolean;
  /** CSS properties and attributes the reveal had to override, deduplicated. */
  changed: string[];
  rect: RevealRect | null;
  png: Uint8Array | null;
  error?: string;
};

const publicHosts = new Map<string, Promise<boolean>>();
function isPublicHost(hostname: string) {
  if (isPrivateHostname(hostname)) return Promise.resolve(false);
  if (!publicHosts.has(hostname)) {
    const timeout = new Promise<boolean>((resolve) => setTimeout(resolve, 3_000, false));
    const resolved = lookup(hostname, { all: true, verbatim: true }).then(
      (addresses) =>
        addresses.length > 0 && addresses.every((entry) => isPublicAddress(entry.address)),
      () => false,
    );
    publicHosts.set(hostname, Promise.race([resolved, timeout]));
  }
  return publicHosts.get(hostname)!;
}

/** Runs in the page: force one marked element and its ancestors visible; report what changed. */
function revealInPage([attribute, marker]: [string, string]) {
  const target = document.querySelector(
    `[${attribute}="${CSS.escape(marker)}"]`,
  ) as HTMLElement | null;
  if (!target) return { found: false, changed: [] as string[], rect: null };
  const changed = new Set<string>();
  const set = (element: HTMLElement, property: string, value: string) => {
    element.style.setProperty(property, value, "important");
    changed.add(property);
  };
  const chain: HTMLElement[] = [];
  for (
    let element: HTMLElement | null = target;
    element && element !== document.documentElement;
    element = element.parentElement
  )
    chain.push(element);
  for (const element of chain) {
    if (element.hasAttribute("hidden")) {
      element.removeAttribute("hidden");
      changed.add("hidden-attribute");
    }
    if (element.tagName === "DETAILS" && !element.hasAttribute("open")) {
      element.setAttribute("open", "");
      changed.add("details-open");
    }
    if (element.tagName === "DIALOG" && !element.hasAttribute("open")) {
      element.setAttribute("open", "");
      changed.add("dialog-open");
    }
    const style = getComputedStyle(element);
    if (style.display === "none")
      set(element, "display", element.tagName === "LI" ? "list-item" : "block");
    if (style.visibility !== "visible") set(element, "visibility", "visible");
    if (Number(style.opacity) < 0.05) set(element, "opacity", "1");
    if (style.contentVisibility === "hidden") set(element, "content-visibility", "visible");
    if (style.clipPath && style.clipPath !== "none") set(element, "clip-path", "none");
    if (style.clip && style.clip !== "auto") set(element, "clip", "auto");
    if (style.transform && style.transform !== "none") set(element, "transform", "none");
    const box = element.getBoundingClientRect();
    if (box.height < 2 || parseFloat(style.maxHeight) === 0) {
      set(element, "max-height", "none");
      set(element, "height", "auto");
    }
    if (box.width < 2 || parseFloat(style.maxWidth) === 0) {
      set(element, "max-width", "none");
      set(element, "width", "auto");
    }
    if (style.overflow !== "visible") set(element, "overflow", "visible");
    // Fixed or absolute boxes parked off-screen (left: -9999px, top: 100vh) come back into the flow.
    const offscreen = box.right <= 0 || box.bottom <= 0 || box.left >= innerWidth * 2;
    if ((style.position === "fixed" || style.position === "absolute") && offscreen)
      set(element, "position", "static");
  }
  const box = target.getBoundingClientRect();
  return {
    found: true,
    changed: [...changed],
    rect: { x: box.x + scrollX, y: box.y + scrollY, width: box.width, height: box.height },
  };
}

export async function revealTarget(
  browser: Browser,
  target: { html: string; url: string; marker: string },
  options: { scripts: boolean; allowData?: boolean },
): Promise<RevealResult> {
  const documentUrl = new URL(target.url);
  documentUrl.hash = "";
  if (!/^https?:$/.test(documentUrl.protocol) || isPrivateHostname(documentUrl.hostname))
    throw new Error("Replay URL must be a public http(s) URL");
  const egress = await replayEgressGuard();
  const context = await browser.newContext({
    viewport: { width: VIEWPORT.width, height: VIEWPORT.height },
    deviceScaleFactor: VIEWPORT.deviceScaleFactor,
    serviceWorkers: "block",
    proxy: { server: egress.server },
    userAgent: "SquirrelScan DOM labeling replay/1.0",
  });
  const timer = setTimeout(() => void context.close().catch(() => undefined), PAGE_DEADLINE_MS);
  const result: RevealResult = {
    marker: target.marker,
    found: false,
    revealed: false,
    changed: [],
    rect: null,
    png: null,
  };
  try {
    await context.routeWebSocket(/.*/, (socket) => socket.close());
    const page = await context.newPage();
    context.on("page", (opened) => {
      if (opened !== page) void opened.close().catch(() => undefined);
    });
    let served = false;
    await context.route("**/*", async (route: Route) => {
      const request = route.request();
      const url = new URL(request.url());
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
        if (!served && url.href.split("#")[0] === documentUrl.href) {
          served = true;
          return route.fulfill({
            status: 200,
            contentType: "text/html; charset=utf-8",
            headers: options.scripts ? {} : { "content-security-policy": "script-src 'none'" },
            body: options.scripts ? target.html : promoteDeferredStyles(target.html).html,
          });
        }
        return route.fulfill({ status: 204 });
      }
      const type = request.resourceType();
      if (
        ALWAYS_BLOCKED.has(type) ||
        (DATA_TYPES.has(type) && !(options.allowData && options.scripts)) ||
        !/^https?:$/.test(url.protocol) ||
        !(await isPublicHost(url.hostname))
      )
        return route.abort("blockedbyclient");
      return route.continue();
    });
    page.setDefaultTimeout(15_000);
    await page.goto(documentUrl.href, { waitUntil: "domcontentloaded", timeout: 20_000 });
    await page.waitForLoadState("networkidle", { timeout: 6_000 }).catch(() => undefined);
    await page.addStyleTag({
      content:
        "*,*::before,*::after{animation:none!important;transition:none!important;caret-color:transparent!important}",
    });
    const revealed = await page.evaluate(revealInPage, [REPLAY_MARKER, target.marker] as [
      string,
      string,
    ]);
    result.found = revealed.found;
    result.changed = revealed.changed;
    result.rect = revealed.rect;
    if (!revealed.rect || revealed.rect.width < 8 || revealed.rect.height < 8) return result;
    result.revealed = true;
    await page.waitForTimeout(300);
    const rect = revealed.rect;
    const x = Math.max(0, rect.x - PAD);
    const y = Math.max(0, rect.y - PAD);
    result.png = await page.screenshot({
      fullPage: true,
      clip: {
        x,
        y,
        width: Math.min(rect.width + 2 * PAD, 2_560),
        height: Math.min(rect.height + 2 * PAD, MAX_CLIP_HEIGHT),
      },
      animations: "disabled",
      timeout: 15_000,
    });
    return result;
  } catch (error) {
    result.error = error instanceof Error ? error.message.slice(0, 200) : String(error);
    return result;
  } finally {
    clearTimeout(timer);
    await context.close().catch(() => undefined);
  }
}
