// Site platform from raw HTML (#2307), the TypeScript twin of
// corpus/cc_corpus.py `platform_of` + `cv_slices.platform_bucket`.
//
// Cheap on purpose: lower-cased substring checks over the first 300k code
// points, plus <meta name="generator"> tags in the first 64k. The runtime uses
// the bucket to apply per-platform threshold overrides (the platform guard);
// the same function tagged every corpus page the guard was computed on, so
// the two must agree (experiments/platform-parity.ts checks it on val pages).

/** First matching platform wins, except that WooCommerce needs WordPress too. */
export const PLATFORM_MARKERS: ReadonlyArray<readonly [string, readonly string[]]> = [
  ["shopify", ["cdn.shopify.com", "shopify.theme", "shopify-section", "myshopify.com"]],
  ["woocommerce", ["woocommerce"]],
  ["wordpress", ["/wp-content/", "/wp-includes/"]],
  ["wix", ["static.wixstatic.com", "wix-bolt"]],
  ["squarespace", ["static1.squarespace.com", "squarespace-cdn.com", "<!-- this is squarespace"]],
  ["webflow", ["data-wf-page", "data-wf-site", "webflow.js", "webflow.com/css"]],
  ["drupal", ["drupal.settings", "drupal-settings-json", "/sites/default/files/"]],
  ["magento", ["magento_", "mage/cookies", "data-mage-init"]],
  ["bigcommerce", [".bigcommerce.com/", "data-stencil", "stencil-utils"]],
  ["ghost", ["ghost-portal", "ghost-search"]],
  ["hubspot", ["hs-sites.com", "hubspotusercontent"]],
  ["joomla", ["/media/jui/"]],
  ["framer", ["framerusercontent.com", "data-framer-"]],
  ["nextjs", ["__next_data__", "/_next/static/"]],
  ["nuxt", ["__nuxt__", "/_nuxt/"]],
  ["astro", ["astro-island", "/_astro/"]],
  ["gatsby", ["___gatsby"]],
];
const GENERATOR = /<meta[^>]+name=["']?generator[^>]*>/gi;
/** In the Python dict's order. */
const GENERATOR_NAMES: ReadonlyArray<readonly [string, string]> = [
  ["wordpress", "wordpress"],
  ["wix.com", "wix"],
  ["webflow", "webflow"],
  ["drupal", "drupal"],
  ["ghost", "ghost"],
  ["hubspot", "hubspot"],
  ["joomla", "joomla"],
  ["squarespace", "squarespace"],
];
/** The platforms the guard and the reports name; everything else is "custom". */
export const PLATFORM_BUCKETS = [
  "shopify",
  "woocommerce",
  "wordpress",
  "wix",
  "squarespace",
  "webflow",
  "nextjs",
] as const;

/** Python's `s[:n]` counts code points; a JS slice counts UTF-16 units. */
function codePointPrefix(text: string, n: number): string {
  if (text.length <= n) return text;
  let units = 0;
  for (let points = 0; points < n && units < text.length; points++) {
    const code = text.charCodeAt(units);
    units += code >= 0xd800 && code <= 0xdbff && units + 1 < text.length ? 2 : 1;
  }
  return text.slice(0, units);
}

export function platformOf(html: string): string {
  const head = codePointPrefix(html, 300_000).toLowerCase();
  const hits = new Set<string>();
  for (const [name, markers] of PLATFORM_MARKERS)
    if (markers.some((marker) => head.includes(marker))) hits.add(name);
  for (const tag of codePointPrefix(head, 65_536).match(GENERATOR) ?? [])
    for (const [needle, name] of GENERATOR_NAMES) if (tag.includes(needle)) hits.add(name);
  if (hits.has("shopify")) return "shopify";
  if (hits.has("woocommerce") && hits.has("wordpress")) return "woocommerce";
  for (const [name] of PLATFORM_MARKERS) if (hits.has(name) && name !== "woocommerce") return name;
  return "custom";
}

export function platformBucket(platform: string | null | undefined): string {
  return platform && (PLATFORM_BUCKETS as readonly string[]).includes(platform)
    ? platform
    : "custom";
}
