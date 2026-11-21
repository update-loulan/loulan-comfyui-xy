"""Local results gallery. All user-selected images are addressed by manifest row."""
import asyncio
import json
import math
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from aiohttp import web


PREFIX = "/loulan-xy"
RUN_NAME = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]{0,159}")
VARIANT = re.compile(r"(?:detail|grid)-[a-f0-9]{32}\.png")
PAGE = Path(__file__).parent / "web" / "review.html"


def child(parent, name):
    if (not isinstance(name, str) or not name or name in (".", "..")
            or any(ch in name for ch in '/\\:\0') or name.endswith((".", " "))):
        raise web.HTTPBadRequest(text="无效的文件名")
    path = parent / name
    if (path.resolve().parent != parent.resolve() or path.is_symlink()
            or getattr(path, "is_junction", lambda: False)()):
        raise web.HTTPBadRequest(text="路径超出测试目录")
    return path


def root_dir():
    import folder_paths
    return child(Path(folder_paths.get_output_directory()).resolve(), "loulan-xy")


def run_dir(request):
    name = request.match_info["run"]
    if not RUN_NAME.fullmatch(name):
        raise web.HTTPBadRequest(text="无效的测试 ID")
    path = child(root_dir(), name)
    if not path.is_dir():
        raise web.HTTPNotFound(text="找不到这次测试")
    return path


def read_json(path):
    if not path.is_file():
        raise web.HTTPNotFound(text="找不到结果文件")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise web.HTTPConflict(text="结果文件暂时无法读取") from exc


def read_manifest(directory):
    data = read_json(child(directory, "manifest.json"))
    if not isinstance(data, dict) or not isinstance(data.get("rows"), list):
        raise web.HTTPConflict(text="测试清单格式有误")
    return data


def source_image(directory, row):
    name = row.get("filename", "")
    parts = name.split("/") if isinstance(name, str) else []
    if len(parts) != 2 or parts[0] != "images" or not parts[1].lower().endswith(".png"):
        raise web.HTTPBadRequest(text="无效的单图路径")
    return child(child(directory, "images"), parts[1])


def allowed_pairs(manifest):
    return {(r["sampler"], r["scheduler"]) for r in manifest["rows"] if r.get("status") == "ok"}


def check_pairs(values, manifest):
    allowed = allowed_pairs(manifest)
    if not isinstance(values, list) or len(values) > len(allowed):
        raise web.HTTPBadRequest(text="组合列表有误")
    pairs = []
    for item in values:
        if (not isinstance(item, list) or len(item) != 2
                or not all(isinstance(s, str) for s in item) or tuple(item) not in allowed):
            raise web.HTTPBadRequest(text="请选择本次测试成功的组合")
        if item not in pairs:
            pairs.append(item)
    return pairs


async def body(request):
    origin = request.headers.get("Origin")
    if origin and urlsplit(origin).netloc.lower() != request.host.lower():
        raise web.HTTPForbidden(text="请求来源与查看器不匹配")
    if request.content_type != "application/json":
        raise web.HTTPUnsupportedMediaType(text="请求需要 JSON")
    try:
        data = await request.json()
    except (ValueError, UnicodeError) as exc:
        raise web.HTTPBadRequest(text="无效的 JSON") from exc
    if not isinstance(data, dict):
        raise web.HTTPBadRequest(text="无效的请求")
    return data


async def page(request):
    return web.FileResponse(PAGE, headers={"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"})


async def sessions(request):
    root = root_dir()
    result = []
    for directory in root.iterdir() if root.is_dir() else ():
        if not RUN_NAME.fullmatch(directory.name):
            continue
        try:
            path = child(child(root, directory.name), "manifest.json")
            if path.is_file():
                result.append((path.stat().st_mtime_ns, directory.name))
        except (OSError, web.HTTPException):
            continue
    return web.json_response([name for _, name in sorted(result, reverse=True)])


async def manifest_view(request):
    directory = run_dir(request)
    data = read_manifest(directory)
    favorites = child(directory, "favorites.json")
    variants = child(directory, "derived")
    return web.json_response({
        "id": directory.name, "status": data.get("status", ""), "settings": data.get("settings", {}),
        "baseline": data.get("baseline"), "rows": data["rows"],
        "has_grid": child(directory, "grid.png").is_file(),
        "favorites": read_json(favorites) if favorites.is_file() else [],
        "variants": sorted((f.name for f in variants.iterdir() if VARIANT.fullmatch(f.name)), reverse=True)
        if variants.is_dir() else [],
    }, headers={"Cache-Control": "no-store"})


async def original_grid(request):
    path = child(run_dir(request), "grid.png")
    if not path.is_file():
        raise web.HTTPNotFound(text="原始整图尚未生成")
    return web.FileResponse(path)


async def image_view(request):
    directory = run_dir(request)
    rows = read_manifest(directory)["rows"]
    token = request.match_info["index"]
    if not re.fullmatch(r"[0-9]{1,9}", token) or int(token) >= len(rows):
        raise web.HTTPNotFound(text="找不到该图格")
    row = rows[int(token)]
    if row.get("status") != "ok":
        raise web.HTTPNotFound(text="该图格没有图片")
    path = source_image(directory, row)
    if not path.is_file():
        raise web.HTTPNotFound(text="单图文件不存在")
    return web.FileResponse(path)


async def save_favorites(request):
    data = await body(request)
    directory = run_dir(request)
    pairs = check_pairs(data.get("pairs"), read_manifest(directory))
    target = child(directory, "favorites.json")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, delete=False) as file:
        temp = Path(file.name)
        try:
            json.dump(pairs, file, ensure_ascii=False)
            file.flush()
        except BaseException:
            file.close()
            temp.unlink(missing_ok=True)
            raise
    try:
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)
    return web.json_response({"favorites": pairs})


def render_variant(directory, data):
    from .render import render_saved_grid

    manifest = read_manifest(directory)
    pairs = check_pairs(data["pairs"], manifest) if data.get("pairs") is not None else None
    if pairs == []:
        raise web.HTTPBadRequest(text="请先选中至少一个组合")
    tile = data.get("tile_size", 256)
    if type(tile) is not int or tile not in (128, 192, 256, 384, 512, 768):
        raise web.HTTPBadRequest(text="无效的图格尺寸")
    transpose = data.get("transpose", False)
    if not isinstance(transpose, bool):
        raise web.HTTPBadRequest(text="无效的排版模式")
    crop = data.get("crop")
    if crop is not None:
        if (not isinstance(crop, list) or len(crop) != 4 or
                not all(type(n) in (int, float) and math.isfinite(n) and 0 <= n <= 1 for n in crop)):
            raise web.HTTPBadRequest(text="无效的细节区域")
        x, y, w, h = crop
        if min(w, h) <= 0 or x + w > 1.000001 or y + h > 1.000001:
            raise web.HTTPBadRequest(text="细节区域超出图像范围")
    for row in manifest["rows"]:
        if row.get("filename"):
            source_image(directory, row)
    if not allowed_pairs(manifest):
        raise web.HTTPBadRequest(text="本次测试还没有成功的图片")
    image = render_saved_grid(directory, manifest, transpose=transpose, tile_size=tile, crop=crop, pairs=pairs)
    folder = child(directory, "derived")
    folder.mkdir(exist_ok=True)
    name = ("detail" if crop else "grid") + "-" + uuid4().hex + ".png"
    target = child(folder, name)
    try:
        with target.open("xb") as stream:
            try:
                image.save(stream, format="PNG")
            except BaseException:
                stream.close()
                target.unlink(missing_ok=True)
                raise
    finally:
        image.close()
    return name


_render_lock = asyncio.Lock()


async def save_variant(request):
    data = await body(request)
    directory = run_dir(request)
    if _render_lock.locked():
        raise web.HTTPConflict(text="另一张对比图正在排版，请稍后再试")
    async with _render_lock:
        try:
            name = await asyncio.to_thread(render_variant, directory, data)
        except (ValueError, FileNotFoundError) as exc:
            raise web.HTTPBadRequest(text=str(exc)) from exc
    return web.json_response({"name": name})


async def variant_view(request):
    directory = run_dir(request)
    name = request.match_info["name"]
    if not VARIANT.fullmatch(name):
        raise web.HTTPNotFound()
    path = child(child(directory, "derived"), name)
    if not path.is_file():
        raise web.HTTPNotFound()
    return web.FileResponse(path)


def register(routes):
    routes.get(PREFIX + "/review")(page)
    routes.get(PREFIX + "/runs")(sessions)
    base = PREFIX + "/runs/{run}"
    routes.get(base)(manifest_view)
    routes.get(base + "/grid")(original_grid)
    routes.get(base + "/image/{index}")(image_view)
    routes.post(base + "/favorites")(save_favorites)
    routes.post(base + "/variants")(save_variant)
    routes.get(base + "/variants/{name}")(variant_view)


try:
    from server import PromptServer
except ImportError:  # Allows gallery tests to supply a standalone aiohttp application.
    PromptServer = None
if getattr(PromptServer, "instance", None) is not None:
    register(PromptServer.instance.routes)
