"""Pillow rendering shared by the XY sampler and the saved-run viewer."""

from functools import lru_cache
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BG = "#171b20"
PANEL = "#252b31"
TEXT = "#f1f4f6"
DIM = "#c2ced5"
ACCENT = "#8edccf"
ERROR = "#ffb1a4"
MAX_GRID_PIXELS = 14_000_000
MAX_GRID_SIDE = 30_000
SETTINGS_HEIGHT = 54


@lru_cache(maxsize=64)
def _font(size):
    for name in ("msyh.ttc", "NotoSansCJK-Regular.ttc", "DejaVuSans.ttf",
                 "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _split_line(draw, value, font, max_width):
    """Split even unbroken extension identifiers without truncating their names."""
    value = str(value)
    lines = []
    while value:
        if draw.textlength(value, font=font) <= max_width:
            lines.append(value)
            break
        lo, hi = 1, len(value)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if draw.textlength(value[:mid], font=font) <= max_width:
                lo = mid
            else:
                hi = mid - 1
        lines.append(value[:lo])
        value = value[lo:]
    return lines


def format_time(row):
    seconds = row.get("seconds")
    if row.get("status") != "ok" or seconds is None:
        return "Failed" if row.get("status") == "error" else "Pending"
    value = f"{seconds:.2f} s"
    ratio = row.get("relative_speed")
    if ratio is not None:
        value += f"  ·  {ratio:.2f}×"
    return value


def annotate_single(image, row):
    """Keep the original pixel rectangle intact and append a measured footer."""
    width, height = image.size
    size = max(9, min(32, width // 37))
    font = _font(size)
    pad = max(2, size // 2)
    measure = ImageDraw.Draw(image)
    lines = []
    for label in (f"{row['sampler']}  /  {row['scheduler']}", format_time(row)):
        lines.extend(_split_line(measure, label, font, max(1, width - 2 * pad)))
    line_h = math.ceil(size * 1.48)
    result = Image.new("RGB", (width, height + line_h * len(lines) + 2 * pad), BG)
    result.paste(image, (0, 0))
    draw = ImageDraw.Draw(result)
    for index, line in enumerate(lines):
        draw.text((width // 2, height + pad + line_h * index + line_h // 2),
                  line, font=font, anchor="mm", fill=TEXT)
    return result


def image_path(directory, row):
    images = (directory / "images").resolve()
    path = (directory / row["filename"]).resolve()
    if images.parent != directory.resolve() or path.parent != images or path.suffix.lower() != ".png":
        raise ValueError("Invalid XY result image path.")
    return path


def _crop_box(width, height, crop):
    if crop is None:
        return (0, 0, width, height)
    if len(crop) != 4 or not all(math.isfinite(float(v)) for v in crop):
        raise ValueError("Crop needs four finite coordinates: x, y, width, height.")
    x, y, w, h = (float(v) for v in crop)
    if not (0 <= x < 1 and 0 <= y < 1 and w > 0 and h > 0):
        raise ValueError("Crop origin or extent is invalid.")
    left, top = min(width - 1, int(x * width)), min(height - 1, int(y * height))
    return (left, top, max(left + 1, min(width, math.ceil((x + w) * width))),
            max(top + 1, min(height, math.ceil((y + h) * height))))


def _cell_text(draw, box, text, color, font_size, background=None, cache=None):
    x, y, width, height = box
    if background:
        draw.rectangle((x, y, x + width - 1, y + height - 1), fill=background)
    cache_key = (width, height, text, font_size)
    layout = cache.get(cache_key) if cache is not None else None
    if layout is None:
        for size in range(font_size, 4, -1):
            font = _font(size)
            lines = []
            for piece in str(text).splitlines():
                lines.extend(_split_line(draw, piece, font, max(1, width - 4)))
            step = max(6, math.ceil(size * 1.3))
            if len(lines) * step <= height - 2:
                break
        visible = max(1, height // step)
        if len(lines) > visible:
            lines = lines[:visible]
            lines[-1] = lines[-1][:-1] + "…"
        layout = (font, lines, step)
        if cache is not None:
            cache[cache_key] = layout
    font, lines, step = layout
    top = y + (height - len(lines) * step) // 2
    for index, line in enumerate(lines):
        draw.text((x + width // 2, top + index * step + step // 2), line,
                  font=font, anchor="mm", fill=color)


def _geometry(nx, ny, requested, aspect):
    def dimensions(side):
        width = max(8, int(side * min(1., aspect)))
        height = max(8, int(side * min(1., 1. / aspect)))
        # Very tall images still need a legible column for names and timing.
        cw = max(width, min(side, 100))
        footer = max(20, min(70, int(cw * .29)))
        head, left = max(28, min(84, cw // 3)), max(54, min(172, cw))
        return (cw, height, footer, head, left,
                left + nx * (cw + 2) + 2, SETTINGS_HEIGHT + head + ny * (height + footer + 2) + 2)
    low, high = 8, max(8, min(4096, int(requested)))
    while low < high:
        mid = (low + high + 1) // 2
        w, h = dimensions(mid)[-2:]
        if w * h <= MAX_GRID_PIXELS and max(w, h) <= MAX_GRID_SIDE:
            low = mid
        else:
            high = mid - 1
    result = dimensions(low)
    w, h = result[-2:]
    if w * h > MAX_GRID_PIXELS or max(w, h) > MAX_GRID_SIDE:
        raise ValueError("This layout exceeds the grid size budget. Select fewer pairs or seeds in the viewer.")
    return result


def render_saved_grid(directory: Path, manifest: dict, *, transpose=False, tile_size=256,
                      crop=None, pairs=None, check_cancel=None):
    settings = manifest["settings"]
    planned = list(dict.fromkeys(tuple(p) for p in settings["pairs"]))
    if pairs is not None:
        selected = {tuple(p) for p in pairs}
        planned = [p for p in planned if p in selected]
    if not planned:
        raise ValueError("The selected grid has no sampler/scheduler pairs.")
    seeds = list(dict.fromkeys(str(s) for s in settings["seeds"]))
    use_pairs = settings.get("paired", False) or len(seeds) > 1
    if use_pairs:
        xs, ys = planned, seeds
        xlabels = [f"{s}\n{t}" for s, t in xs]
        ylabels = [f"Seed\n{s}" for s in ys]
    else:
        xs = list(dict.fromkeys(s for s, _ in planned))
        ys = list(dict.fromkeys(t for _, t in planned))
        xlabels, ylabels = xs, ys

    rows = {(r["sampler"], r["scheduler"], str(r["seed"])): r for r in manifest["rows"]}
    ok = next((r for r in manifest["rows"] if r["status"] == "ok"), None)
    aspect = 1.
    if ok:
        l, t, r, b = _crop_box(ok["width"], ok["height"], crop)
        aspect = (r - l) / (b - t)
    if transpose:
        xlabels, ylabels = ylabels, xlabels
    cw, ph, foot, head, left, gw, gh = _geometry(len(xlabels), len(ylabels), tile_size, aspect)
    grid = Image.new("RGB", (gw, gh), BG)
    draw = ImageDraw.Draw(grid)
    cache = {}
    font_size = min(20, max(6, cw // 12))

    seed_label = ("Seed " + seeds[0] if len(seeds) == 1 else
                  f"Seeds ({len(seeds)}) " + ", ".join(seeds[:3]) + ("…" if len(seeds) > 3 else ""))
    header = "   ·   ".join([seed_label, *(f"{label} {settings.get(key, '—')}"
                            for key, label in (("steps", "Steps"), ("cfg", "CFG"), ("denoise", "Denoise")))])
    draw.rectangle((0, 0, gw - 1, SETTINGS_HEIGHT - 2), fill=PANEL)
    header_width = min(gw, int(draw.textlength(header, font=_font(17))) + 32)
    _cell_text(draw, (0, 0, header_width, SETTINGS_HEIGHT - 2), header, TEXT, 17)
    for i, label in enumerate(xlabels):
        _cell_text(draw, (left + i * (cw + 2), SETTINGS_HEIGHT, cw, head), label, ACCENT,
                   font_size, cache=cache)
    first_row_y = SETTINGS_HEIGHT + head
    for i, label in enumerate(ylabels):
        _cell_text(draw, (0, first_row_y + i * (ph + foot + 2), left - 3, ph + foot),
                   label, ACCENT, min(16, font_size), cache=cache)
    selected = set(planned)
    for yi, yvalue in enumerate(ys):
        for xi, xvalue in enumerate(xs):
            if check_cancel is not None:
                check_cancel()
            if use_pairs:
                (sampler, scheduler), seed = xvalue, yvalue
            else:
                sampler, scheduler, seed = xvalue, yvalue, seeds[0]
            if (sampler, scheduler) not in selected:
                continue
            gx, gy = (yi, xi) if transpose else (xi, yi)
            dx, dy = left + gx * (cw + 2), first_row_y + gy * (ph + foot + 2)
            row = rows.get((sampler, scheduler, seed), {})
            status = row.get("status", "pending")
            problem = status != "ok"
            if not problem:
                try:
                    with Image.open(image_path(directory, row)) as source:
                        box = _crop_box(row["width"], row["height"], crop)
                        view = source.crop(box)
                    view.thumbnail((cw, ph), Image.Resampling.LANCZOS)
                    grid.paste(view, (dx + (cw - view.width) // 2, dy + (ph - view.height) // 2))
                    view.close()
                except (OSError, FileNotFoundError):
                    problem, status = True, "missing image"
            if problem:
                message = status.upper()
                if row.get("error"):
                    reason = " ".join(str(row["error"]).split())
                    message += "\n" + (reason if len(reason) <= 84 else reason[:83] + "…")
                _cell_text(draw, (dx, dy, cw, ph), message, ERROR, font_size, PANEL, cache)
            caption = f"{sampler} / {scheduler}\n{format_time(row)}"
            _cell_text(draw, (dx, dy + ph, cw, foot), caption, DIM, font_size, PANEL, cache)
    return grid
