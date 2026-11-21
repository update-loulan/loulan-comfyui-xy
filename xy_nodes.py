import hashlib
import json
import re
from pathlib import Path

import comfy.samplers
import folder_paths
import torch
from comfy_api.latest import io
from comfy_execution.graph_utils import ExecutionBlocker, is_link

from .engine import run_matrix


MODEL_INPUTS = ("model", "positive", "negative", "latent", "vae")
MODEL_FILES = {
    "ckpt_name": "checkpoints", "unet_name": "diffusion_models",
    "clip_name": "text_encoders", "clip_name1": "text_encoders",
    "clip_name2": "text_encoders", "clip_name3": "text_encoders",
    "vae_name": "vae", "lora_name": "loras", "control_net_name": "controlnet",
}


def source_identity(hidden, latent, positive, negative):
    """Fingerprint only the five generation inputs, independent of the UI and XY settings."""
    digest = hashlib.sha256()
    visited = set()

    def add(value):
        if isinstance(value, torch.Tensor):
            digest.update(str((tuple(value.shape), value.dtype)).encode())
            raw = value.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8)
            digest.update(memoryview(raw.numpy()))
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                add(str(key))
                add(value[key])
        elif isinstance(value, (tuple, list)):
            digest.update(b"[")
            for item in value:
                add(item)
            digest.update(b"]")
        else:
            token = value if isinstance(value, (str, int, float, bool, type(None))) else type(value).__qualname__
            digest.update(json.dumps(token, ensure_ascii=False).encode() + b"\0")

    def visit(node_id):
        if node_id in visited:
            return
        visited.add(node_id)
        node = hidden.dynprompt.get_node(node_id)
        add([node_id, node["class_type"], node["inputs"]])
        for key, value in node["inputs"].items():
            if is_link(value):
                visit(value[0])
            elif key in MODEL_FILES and isinstance(value, str):
                filename = folder_paths.get_full_path(MODEL_FILES[key], value)
                if filename and Path(filename).is_file():
                    stat = Path(filename).stat()
                    add([filename, stat.st_size, stat.st_mtime_ns])

    inputs = hidden.dynprompt.get_node(hidden.unique_id)["inputs"]
    for name in MODEL_INPUTS:
        if is_link(inputs.get(name)):
            add([name, inputs[name]])
            visit(inputs[name][0])
    for data in (latent, positive, negative):
        add(data)
    return digest.hexdigest()


def check_names(names, available, label):
    if not isinstance(names, (tuple, list)) or not all(isinstance(n, str) for n in names):
        raise ValueError(f"{label}选择格式错误。")
    missing = [n for n in names if n not in available]
    if missing:
        raise ValueError(f"当前 KSampler 缺少{label}：{', '.join(missing)}。请重新选择或恢复对应插件。")
    return list(dict.fromkeys(names))


def parse_seeds(value, default):
    fields = re.split(r"[,，;；\s]+", value.strip()) if value.strip() else [str(default)]
    if not all(s.isascii() and s.isdecimal() and int(s) < 2**64 for s in fields):
        raise ValueError("复测种子必须是 0 至 2⁶⁴−1 的整数，用逗号或空格分隔。")
    return list(dict.fromkeys(map(int, fields)))


def make_pairs(samplers, schedulers, favorites, baseline):
    explicit = json.loads(favorites) if favorites.strip() else []
    if not isinstance(explicit, list) or any(
        not isinstance(p, list) or len(p) != 2 or not all(isinstance(n, str) for n in p)
        for p in explicit
    ):
        raise ValueError('收藏配对格式为 [["euler","simple"],["heun","karras"]]。')
    pairs = [tuple(p) for p in explicit] or [(s, d) for d in schedulers for s in samplers]
    if not pairs:
        raise ValueError("至少选一个采样器和一个调度器。")
    if baseline:
        pairs = [baseline, *pairs]
    pairs = list(dict.fromkeys(pairs))
    check_names([p[0] for p in pairs], comfy.samplers.KSampler.SAMPLERS, "采样器")
    check_names([p[1] for p in pairs], comfy.samplers.KSampler.SCHEDULERS, "调度器")
    return pairs, bool(explicit)


class _XYNode(io.ComfyNode):
    sampler_axis = True

    @classmethod
    def define_schema(cls):
        samplers = list(comfy.samplers.KSampler.SAMPLERS)
        schedules = list(comfy.samplers.KSampler.SCHEDULERS)
        own, opposite = (samplers, schedules) if cls.sampler_axis else (schedules, samplers)
        own_type, other_type = ("LOULAN_SAMPLERS", "LOULAN_SCHEDULERS") if cls.sampler_axis else ("LOULAN_SCHEDULERS", "LOULAN_SAMPLERS")
        name = "采样器" if cls.sampler_axis else "调度器"
        other = "调度器" if cls.sampler_axis else "采样器"
        return io.Schema(
            node_id="LoulanXYSampler" if cls.sampler_axis else "LoulanXYScheduler",
            display_name=f"loulan-xy · {name}测试", category="loulan-xy",
            description="test 执行测试；selection 只输出选择列表。点击节点上的结果按钮可查看、收藏和复测。",
            is_output_node=True, hidden=[io.Hidden.unique_id, io.Hidden.dynprompt],
            inputs=[
                io.Combo.Input("mode", options=["test", "selection"], display_name="用途", default="test"),
                io.MultiCombo.Input("selection", options=own, display_name=name, default=[own[0]], chip=True),
                io.Combo.Input("fixed", options=opposite, display_name=f"固定{other}", default=opposite[0]),
                io.Custom(other_type).Input("other_axis", optional=True, display_name=f"{other}列表"),
                io.Model.Input("model", optional=True), io.Conditioning.Input("positive", optional=True),
                io.Conditioning.Input("negative", optional=True), io.Latent.Input("latent", optional=True),
                io.Vae.Input("vae", optional=True),
                io.Int.Input("seed", default=42, min=0, max=2**64-1),
                io.Int.Input("steps", default=20, min=1, max=10000),
                io.Float.Input("cfg", default=4.0, min=0, max=100, step=0.1),
                io.Float.Input("denoise", default=1.0, min=0, max=1, step=0.01),
                io.String.Input("session", default="loulan-xy", display_name="测试名称"),
                io.Boolean.Input("resume", default=False, display_name="续跑同配置 / 只补缺图"),
                io.Int.Input("tile_size", default=320, min=64, max=4096, step=16, display_name="总图格子最长边"),
                io.Boolean.Input("transpose", default=False, display_name="交换横纵轴"),
                io.Boolean.Input("use_baseline", default=False, display_name="加入基准组合"),
                io.Combo.Input("baseline_sampler", options=samplers, default=samplers[0], display_name="基准采样器", advanced=True),
                io.Combo.Input("baseline_scheduler", options=schedules, default=schedules[0], display_name="基准调度器", advanced=True),
                io.String.Input("favorites", default="", display_name="仅测收藏配对 (JSON)", advanced=True),
                io.String.Input("seeds", default="", display_name="复测种子（逗号分隔）", advanced=True),
            ],
            outputs=[io.Custom(own_type).Output(display_name=f"{name}列表"), io.Image.Output(display_name="XY 总图"),
                     io.String.Output(display_name="结果目录名")],
        )

    @classmethod
    def fingerprint_inputs(cls, mode="test", selection=None, **kwargs):
        return float("nan") if mode == "test" else json.dumps(selection)

    @classmethod
    def execute(cls, mode, selection, fixed, seed, steps, cfg, denoise, session, resume, tile_size, transpose,
                use_baseline, baseline_sampler, baseline_scheduler, favorites="", seeds="", other_axis=None,
                model=None, positive=None, negative=None, latent=None, vae=None):
        available = comfy.samplers.KSampler
        own = check_names(selection, available.SAMPLERS if cls.sampler_axis else available.SCHEDULERS,
                          "采样器" if cls.sampler_axis else "调度器")
        if mode == "selection":
            if not own:
                raise ValueError("选择列表为空。")
            return io.NodeOutput(own, ExecutionBlocker(None), ExecutionBlocker(None))
        opposite = other_axis if other_axis is not None else [fixed]
        samplers, schedules = (own, opposite) if cls.sampler_axis else (opposite, own)
        baseline = (baseline_sampler, baseline_scheduler) if use_baseline else None
        pairs, paired = make_pairs(samplers, schedules, favorites, baseline)
        seed_list = parse_seeds(seeds, seed)
        supplied = (model, positive, negative, latent, vae)
        missing = [key for key, value in zip(MODEL_INPUTS, supplied) if value is None]
        if missing:
            raise ValueError(f"测试端需要连接 {', '.join(missing)}。只输出列表请将用途改成 selection。")
        samples = latent["samples"]
        if samples.shape[0] != 1 or not (samples.ndim == 4 or samples.ndim == 5 and samples.shape[2] == 1):
            raise ValueError("loulan-xy 的计时单位为一张静态图片，请使用 batch_size=1 的单帧 Latent。")
        grid, run_id = run_matrix(
            pairs=pairs, seeds=seed_list, model=model, positive=positive, negative=negative, latent=latent, vae=vae,
            steps=steps, cfg=cfg, denoise=denoise, session=session, resume=resume,
            identity=source_identity(cls.hidden, latent, positive, negative),
            prompt=cls.hidden.prompt, extra_pnginfo=cls.hidden.extra_pnginfo, transpose=transpose,
            tile_size=tile_size, baseline=baseline, paired=paired or len(seed_list) > 1,
        )
        preview = {"filename":"grid.png", "subfolder":f"loulan-xy/{run_id}", "type":"output"}
        return io.NodeOutput(own, grid, run_id, ui={"images":[preview], "run_id":[run_id]})


class LoulanXYSampler(_XYNode):
    sampler_axis = True


class LoulanXYScheduler(_XYNode):
    sampler_axis = False
