/**
 * A scripts-off replay serves the document with `script-src 'none'`, which also
 * blocks inline event handlers, while `<noscript>` stays unrendered (scripting
 * is still enabled). Stylesheets deferred by a handler, the loadCSS pattern
 * `rel="preload" as="style" onload="this.rel='stylesheet'"` and the
 * `media="print" onload="this.media='all'"` pattern (both common on Shopify and
 * WordPress), therefore never apply and the page renders unstyled. Promote them
 * to plain stylesheets, for scripts-off replays only.
 */
export function promoteDeferredStyles(html: string) {
  let promoted = 0;
  const out = html.replace(/<link\b[^>]*>/gi, (tag) => {
    if (!/\sonload\s*=/i.test(tag)) return tag;
    let next = tag;
    if (/\sas\s*=\s*["']?style\b/i.test(tag))
      next = next.replace(/\srel\s*=\s*(["']?)preload\1/i, ' rel="stylesheet"');
    if (/this\.media/i.test(tag))
      next = next.replace(/\smedia\s*=\s*(["']?)print\1/i, ' media="all"');
    if (next === tag) return tag;
    promoted++;
    return next.replace(/\sonload\s*=\s*("[^"]*"|'[^']*'|[^\s>]+)/i, "");
  });
  return { html: out, promoted };
}
