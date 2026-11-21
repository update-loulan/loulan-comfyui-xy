"""Run a comparison against the native KSampler and checkpoint results per cell."""

from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import time
from uuid import uuid4

import numpy as np
from PIL import Image, PngImagePlugin
import torch

import comfy.model_management as mm
import comfy.samplers
import comfy.utils
from comfy.cli_args import args
from comfy_execution.utils import get_executing_context
import folder_paths
import nodes
from server import PromptServer

from .render import annotate_single, image_path, render_saved_grid


LOG = logging.getLogger(__name__)
CSV_FIELDS = ("row", "column", "sampler", "scheduler", "seed", "status", "seconds", "relative_speed",
              "relative_time", "width", "height", "filename", "error")


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _slug(value, length):
    return re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-_")[:length] or "xy"


@contextmanager
def _atomic_output(path):
    temporary = path.with_name("." + path.name + "." + uuid4().hex + ".tmp")
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path, data):
    with _atomic_output(path) as tmp:
        with tmp.open("w", encoding="utf-8") as fp:
            json.dump(data, fp, ensure_ascii=False, allow_nan=False, indent=1)


def _checkpoint(directory, manifest, ordered_keys, records):
    manifest["rows"] = [records[key] for key in ordered_keys if key in records]
    manifest["updated"] = _now()
    written = sum(r["status"] == "ok" for r in manifest["rows"])
    manifest["counts"] = {"total": len(ordered_keys), "ok": written,
                          "error": len(manifest["rows"]) - written}
    with _atomic_output(directory / "results.csv") as tmp:
        with tmp.open("w", encoding="utf-8-sig", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(manifest["rows"])
    _write_json(directory / "manifest.json", manifest)


class _Busy(Exception):
    pass


@contextmanager
def _run_lock(directory):
    # OS file locks disappear on process exit, so interrupted runs remain resumable.
    with (directory / ".run.lock").open("a+b") as fp:
        fp.seek(0, os.SEEK_END)
        if fp.tell() == 0:
            fp.write(b"0")
            fp.flush()
        fp.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(fp.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise _Busy() from exc
        try:
            yield
        finally:
            if os.name == "nt":
                fp.seek(0)
                msvcrt.locking(fp.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(fp.fileno(), fcntl.LOCK_UN)


@contextmanager
def _select_run(root, session, signature, resume):
    prefix = _slug(session, 28) + "-" + _digest(session)[:6] + "-"
    root.mkdir(parents=True, exist_ok=True)
    if resume:
        for directory in sorted(root.glob(prefix + "*"), reverse=True):
            if not directory.is_dir() or directory.resolve().parent != root.resolve():
                continue
            path = directory / "manifest.json"
            if not path.exists():
                continue
            try:
                with _run_lock(directory):
                    try:
                        data = json.loads(path.read_text(encoding="utf-8"))
                    except (ValueError, UnicodeError):
                        continue
                    if (isinstance(data, dict) and data.get("id") == directory.name
                            and data.get("signature") == signature and data.get("session") == session):
                        yield directory, data
                        return
            except _Busy:
                continue
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    directory = root / (prefix + stamp + "-" + uuid4().hex[:8])
    directory.mkdir()
    (directory / "images").mkdir()
    with _run_lock(directory):
        yield directory, None


def _sha256(path):
    with path.open("rb") as fp:
        return hashlib.file_digest(fp, "sha256").hexdigest()


def _filename(sampler, scheduler, seed, row, column):
    key = _digest([sampler, scheduler, str(seed)])[:12]
    return (f"images/r{row:02d}_c{column:02d}__{_slug(sampler, 55)}__"
            f"{_slug(scheduler, 32)}__s{seed}__{key}.png")


def _healthy(directory, row):
    try:
        if (row["status"] != "ok" or row["width"] <= 0 or row["height"] <= 0
                or row["filename"] != _filename(row["sampler"], row["scheduler"], row["seed"],
                                                row["row"], row["column"])):
            return False
        path = image_path(directory, row)
        with Image.open(path) as source:
            if source.format != "PNG" or source.width != row["width"] or source.height <= row["height"]:
                return False
        return _sha256(path) == row["sha256"]
    except (TypeError, KeyError, ValueError, OSError):
        return False


def _png_info(prompt, extra_pnginfo, cell):
    if args.disable_metadata:
        return None
    info = PngImagePlugin.PngInfo()
    entries = dict(extra_pnginfo or {})
    if prompt is not None:
        entries["prompt"] = prompt
    entries["loulan_xy"] = cell
    for name, value in entries.items():
        info.add_text(str(name), json.dumps(value, ensure_ascii=False, default=str))
    return info


def _write_png(image, path, metadata):
    with _atomic_output(path) as tmp:
        image.save(tmp, format="PNG", pnginfo=metadata, compress_level=4)


class _UnsupportedImageShape(ValueError):
    pass


def _sample_one(model, positive, negative, latent, vae, sampler, scheduler,
                seed, steps, cfg, denoise):
    mm.synchronize()
    start = time.perf_counter()
    sampled, = nodes.common_ksampler(model, int(seed), steps, cfg, sampler, scheduler,
                                     positive, negative, latent, denoise=denoise)
    mm.synchronize()
    seconds = time.perf_counter() - start
    mm.throw_exception_if_processing_interrupted()
    output = vae.decode(sampled["samples"])
    if output.ndim == 5 and output.shape[:2] == (1, 1):
        output = output[:, 0]
    if output.ndim != 4 or output.shape[0] != 1 or output.shape[-1] not in (3, 4):
        raise _UnsupportedImageShape("Loulan XY requires one decoded RGB/RGBA still image per cell.")
    array = output[0, :, :, :3].detach().to(device="cpu", dtype=torch.float32).numpy()
    if not np.isfinite(array).all():
        raise ValueError("The decoder produced non-finite image pixels.")
    pixels = np.clip(255.0 * array, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels), seconds


def _set_relative(row, reference):
    seconds = row.get("seconds")
    row["relative_speed"] = reference / seconds if reference and seconds else None
    row["relative_time"] = seconds / reference if reference and seconds else None


def _save_cell(directory, original, row, signature, prompt, extra_pnginfo):
    with annotate_single(original, row) as annotated:
        path = image_path(directory, row)
        info = _png_info(prompt, extra_pnginfo, {**row, "signature": signature, "sha256": None})
        _write_png(annotated, path, info)
    row["sha256"] = _sha256(path)


def _publish_cell(run_id, index, total, sampler, scheduler):
    server = getattr(PromptServer, "instance", None)
    if server is None:
        return
    context = get_executing_context()
    node_id = context.node_id if context is not None else getattr(server, "last_node_id", None)
    try:
        server.send_sync("loulan-xy-progress", {
            "run_id": run_id, "index": index, "total": total, "sampler": sampler,
            "scheduler": scheduler, "node_id": node_id,
        }, getattr(server, "client_id", None))
    except (OSError, RuntimeError):
        LOG.debug("Loulan XY progress listener unavailable", exc_info=True)


def run_matrix(*, pairs, seeds, model, positive, negative, latent, vae, steps, cfg,
               denoise, session, resume, identity, prompt, extra_pnginfo, transpose,
               tile_size, baseline, paired):
    """Return the saved overview as a single Comfy IMAGE and its stable run ID."""
    pairs = list(dict.fromkeys(tuple(pair) for pair in pairs))
    seeds = list(dict.fromkeys(str(int(s)) for s in seeds))
    baseline = tuple(baseline) if baseline else None
    if not pairs or not seeds:
        raise ValueError("Select at least one pair and one seed.")
    if any(not 0 <= int(s) < 2**64 for s in seeds):
        raise ValueError("Seeds must be unsigned 64-bit integers.")
    if baseline is not None and baseline not in pairs:
        raise ValueError("The baseline must be one of the selected pairs.")
    native = comfy.samplers.KSampler
    for s, t in pairs:
        if s not in native.SAMPLERS or t not in native.SCHEDULERS:
            raise ValueError(f"Sampler or scheduler no longer available: {s} / {t}")
    samples = latent["samples"]
    if (getattr(samples, "is_nested", False) or samples.ndim not in (4, 5)
            or samples.shape[0] != 1 or (samples.ndim == 5 and samples.shape[2] != 1)):
        raise ValueError("Loulan XY requires one image LATENT [1,C,H,W] or [1,C,1,H,W]. Use seeds for variations.")
    session = str(session or "loulan-xy").strip() or "loulan-xy"
    settings = {"pairs": [list(pair) for pair in pairs], "seeds": seeds, "steps": steps,
                "cfg": cfg, "denoise": denoise, "paired": bool(paired),
                "baseline": list(baseline) if baseline else None}
    signature = _digest({"format": 2, "identity": identity, "settings": settings})
    root = Path(folder_paths.get_output_directory()) / "loulan-xy"
    keys = [(s, t, seed) for seed in seeds for s, t in pairs]
    if paired or len(seeds) > 1:
        columns = {pair: i for i, pair in enumerate(pairs, 1)}
        coordinates = {(s, t, seed): (j, columns[s, t])
                       for j, seed in enumerate(seeds, 1) for s, t in pairs}
    else:
        columns = {s: i for i, s in enumerate(dict.fromkeys(s for s, t in pairs), 1)}
        rows = {t: i for i, t in enumerate(dict.fromkeys(t for s, t in pairs), 1)}
        coordinates = {(s, t, seed): (rows[t], columns[s]) for s, t, seed in keys}
    if baseline:
        # The viewer addresses cells by published row index; process and publish
        # in the same order so appending cells never changes earlier URLs.
        keys.sort(key=lambda key: key[:2] != baseline)
    with _select_run(root, session, signature, resume) as (directory, manifest):
        if manifest is None:
            manifest = {"version": 2, "id": directory.name, "session": session,
                        "created": _now(), "settings": settings, "signature": signature,
                        "baseline": settings["baseline"], "rows": []}
        records = {(r["sampler"], r["scheduler"], str(r["seed"])): r for r in manifest["rows"]}
        manifest.pop("message", None)
        manifest.update(status="running", grid="grid.png", transpose=bool(transpose), tile_size=tile_size)
        reference_times = {}
        total = len(keys)
        progress = comfy.utils.ProgressBar(total)
        last_event = 0.0
        try:
            _checkpoint(directory, manifest, keys, records)
            for index, key in enumerate(keys, start=1):
                mm.throw_exception_if_processing_interrupted()
                sampler, scheduler, seed = key
                now = time.monotonic()
                if index in (1, total) or now - last_event >= .2:
                    _publish_cell(directory.name, index, total, sampler, scheduler)
                    last_event = now
                previous = records.get(key)
                LOG.info("Loulan XY %d/%d: %s / %s, seed %s", index, total, sampler, scheduler, seed)
                if previous and _healthy(directory, previous):
                    if key[:2] == baseline:
                        reference_times[seed] = previous["seconds"]
                    old_ratio = previous.get("relative_speed")
                    _set_relative(previous, reference_times.get(seed))
                    if old_ratio != previous["relative_speed"]:
                        with Image.open(image_path(directory, previous)) as annotated:
                            original = annotated.crop((0, 0, previous["width"], previous["height"]))
                        with original:
                            _save_cell(directory, original, previous, signature, prompt, extra_pnginfo)
                        _checkpoint(directory, manifest, keys, records)
                    progress.update_absolute(index, total)
                    continue
                r, c = coordinates[key]
                row = {"row": r, "column": c, "sampler": sampler, "scheduler": scheduler, "seed": seed,
                       "status": "error", "seconds": None, "width": 0, "height": 0,
                       "filename": "", "relative_time": None, "relative_speed": None}
                try:
                    original, seconds = _sample_one(model, positive, negative, latent, vae,
                                                    sampler, scheduler, seed, steps, cfg, denoise)
                except (MemoryError, OSError, _UnsupportedImageShape, mm.InterruptProcessingException):
                    raise
                except Exception as exc:
                    text = str(exc).lower()
                    if mm.is_oom(exc) or any(marker in text for marker in
                                              ("device-side assert", "out of memory", "not enough memory")):
                        raise
                    row["error"] = f"{type(exc).__name__}: {exc}"[:1800]
                    LOG.warning("Loulan XY failed cell %s / %s: %s", sampler, scheduler, row["error"])
                else:
                    with original:
                        row.update(status="ok", seconds=seconds, width=original.width,
                                   height=original.height, filename=_filename(sampler, scheduler, seed, r, c))
                        if key[:2] == baseline:
                            reference_times[seed] = seconds
                        _set_relative(row, reference_times.get(seed))
                        _save_cell(directory, original, row, signature, prompt, extra_pnginfo)
                records[key] = row
                _checkpoint(directory, manifest, keys, records)
                progress.update_absolute(index, total)
            mm.throw_exception_if_processing_interrupted()
            with render_saved_grid(directory, manifest, transpose=transpose, tile_size=tile_size,
                                   check_cancel=mm.throw_exception_if_processing_interrupted) as grid:
                _write_png(grid, directory / "grid.png", _png_info(prompt, extra_pnginfo,
                           {"run": directory.name, "signature": signature, "settings": settings}))
                mm.throw_exception_if_processing_interrupted()
                tensor = torch.from_numpy(np.array(grid)).to(torch.float32).div_(255).unsqueeze(0)
            manifest["status"] = ("complete_with_errors" if manifest["counts"]["error"] else "complete")
            _checkpoint(directory, manifest, keys, records)
            return tensor, directory.name
        except BaseException as exc:
            manifest["status"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, mm.InterruptProcessingException)) else "stopped"
            manifest["message"] = str(exc)[:1800]
            try:
                _checkpoint(directory, manifest, keys, records)
            except OSError:
                LOG.exception("Loulan XY could not save the last status update.")
            raise
