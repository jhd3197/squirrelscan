"""The adjudicator's views of one capture: page top, overview, outlined target bands (#2307).

Shared by the Modal renderer (modal_render.py) and the local teacher prep
(teacher_prep.py), which cuts the same views from existing labeller captures.

  page.jpg              the top 2400 px of the page at 0.75 scale
  overview.jpg          the whole capture as a narrow thumbnail
  n<N>.jpg              a full-width band around target N, outlined and tagged TARGET
  n<N>-overview.jpg     the thumbnail with target N outlined
"""

from __future__ import annotations

import io

FONTS = [
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
]


def _font():
    from PIL import ImageFont

    for path in FONTS:
        try:
            return ImageFont.truetype(path, 16)
        except OSError:
            continue
    return ImageFont.load_default()


PAGE_TOP = 2400
PAGE_SCALE = 0.75
OVERVIEW_WIDTH = 320
OVERVIEW_MAX_HEIGHT = 2400
BAND_CONTEXT = 250
BAND_MAX = 1700
BAND_SCALE = 0.8
OUTLINE = (0, 200, 83)


def _jpeg(img, quality: int = 82) -> bytes:
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, "JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def _outline(draw, box, width: int, label: bool, font=None):
    x0, y0, x1, y1 = box
    # A dark hairline outside the green keeps the outline visible on green pages.
    draw.rectangle((x0 - width - 1, y0 - width - 1, x1 + width + 1, y1 + width + 1), outline=(20, 20, 20), width=1)
    draw.rectangle((x0 - width, y0 - width, x1 + width, y1 + width), outline=OUTLINE, width=width)
    if label:
        # The tag sits OUTSIDE the box (above, else below) so it never covers the target's own content;
        # only a target filling the whole band gets it inside, at its bottom edge.
        height = draw.im.size[1]
        if y0 - width - 26 >= 0:
            tag_y = y0 - width - 26
        elif y1 + width + 26 <= height:
            tag_y = y1 + width + 2
        else:
            tag_y = max(0, y1 - 26)
        draw.rectangle((x0 - width, tag_y, x0 - width + 84, tag_y + 24), fill=OUTLINE)
        draw.text((x0 - width + 8, tag_y + 4), "TARGET", fill=(255, 255, 255), font=font)


def compose_views(png: bytes, located: dict) -> dict[str, bytes]:
    from PIL import Image, ImageDraw

    Image.MAX_IMAGE_PIXELS = None
    shot = Image.open(io.BytesIO(png)).convert("RGB")
    width, height = shot.size
    font = _font()
    views: dict[str, bytes] = {}
    top = shot.crop((0, 0, width, min(height, PAGE_TOP)))
    views["page"] = _jpeg(top.resize((int(top.width * PAGE_SCALE), int(top.height * PAGE_SCALE)), Image.LANCZOS))
    scale = min(OVERVIEW_WIDTH / width, OVERVIEW_MAX_HEIGHT / height)
    overview = shot.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.LANCZOS)
    views["overview"] = _jpeg(overview, 75)
    for marker, at in located.items():
        rect = at.get("rect")
        if not (at.get("visible") and rect):
            continue
        x0, y0 = rect["x"], rect["y"]
        x1, y1 = x0 + rect["width"], y0 + rect["height"]
        top_y = max(0, y0 - BAND_CONTEXT)
        bottom_y = min(height, y1 + BAND_CONTEXT, top_y + BAND_MAX)
        band = shot.crop((0, top_y, width, bottom_y))
        draw = ImageDraw.Draw(band)
        _outline(draw, (x0, y0 - top_y, x1, min(y1, bottom_y) - top_y), 5, True, font)
        views[f"n{marker}"] = _jpeg(
            band.resize((int(band.width * BAND_SCALE), int(band.height * BAND_SCALE)), Image.LANCZOS)
        )
        thumb = overview.copy()
        tdraw = ImageDraw.Draw(thumb)
        _outline(tdraw, (x0 * scale, y0 * scale, x1 * scale, y1 * scale), 3, False)
        views[f"n{marker}-overview"] = _jpeg(thumb, 75)
    return views
