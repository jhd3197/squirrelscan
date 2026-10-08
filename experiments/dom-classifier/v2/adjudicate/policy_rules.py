"""Page-type policy v8.2's mechanical rules, as one function (#2307).

v8 fixed two cases from the URL alone: the site root is the homepage, and a
docs/article/other page under a /blog/ path is a blog_post. v8.2 adds two more
(drafted as R1/R2 from the adjudicator pilot, decided 2026-10-09):

- rule 8: a locale root (`/de/`, `/en-us/`, `/pt_br/`, `/zh-hant/`; ISO 639-1
  language codes plus `fil`) is that locale's homepage;
- rule 1: Wix's `/post/<slug>` is a blog path, so a docs/article/other page
  there is a blog_post (same label scope as the /blog/ rule).

Then the 12-class fold (v8's six rare classes are `other`).
"""

from __future__ import annotations

import re
from urllib.parse import unquote, urlsplit

PAGE_FOLD = {"event", "search_results", "local_business", "job_posting", "faq", "recipe"}
ISO639_1 = set(
    "aa ab af ak am an ar as av ay az ba be bg bh bi bm bn bo br bs ca ce ch co cr cs cu cv cy da de dv dz ee el "
    "en eo es et eu fa ff fi fj fo fr fy ga gd gl gn gu gv ha he hi ho hr ht hu hy hz ia id ie ig ii ik io is it iu "
    "ja jv ka kg ki kj kk kl km kn ko kr ks ku kv kw ky la lb lg li ln lo lt lu lv mg mh mi mk ml mn mr ms mt my na "
    "nb nd ne ng nl nn no nr nv ny oc oj om or os pa pi pl ps pt qu rm rn ro ru rw sa sc sd se sg si sk sl sm sn so "
    "sq sr ss st su sv sw ta te tg th ti tk tl tn to tr ts tt tw ty ug uk ur uz ve vi vo wa wo xh yi yo za zh zu fil".split()
)
LOCALE = re.compile(r"^([a-z]{2,3})([-_]([a-z]{2}|hans|hant))?$")


def segments(url: str) -> list[str]:
    return [unquote(s).lower() for s in urlsplit(url).path.split("/") if s]


def is_locale_root(url: str) -> bool:
    segs = segments(url)
    if len(segs) != 1 or urlsplit(url).query:
        return False
    match = LOCALE.match(segs[0])
    return bool(match and match.group(1) in ISO639_1)


def is_wix_post(url: str, platform: str | None) -> bool:
    segs = segments(url)
    return platform == "wix" and len(segs) >= 2 and segs[0] == "post"


def page_rules_v82(label: str, url: str, platform: str | None = None) -> str:
    parts = urlsplit(url)
    segs = segments(url)
    fixed = label
    if (not segs and not parts.query) or is_locale_root(url):
        fixed = "homepage"
    elif ("blog" in segs[:-1] or is_wix_post(url, platform)) and label in ("docs", "article", "other"):
        fixed = "blog_post"
    return "other" if fixed in PAGE_FOLD else fixed
