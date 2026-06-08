#!/usr/bin/env python3
"""Qwen3-VL SoM rerouting experiment for THGS ramen object queries.

The experiment is intentionally narrow: it only targets low-IoU object masks in
the LERF ramen scene. It builds numbered segment candidate panels, asks a local
Qwen3-VL model to select candidate ids, applies the selected masks to a copy of
the THGS baseline predictions, and writes diagnostics for post-hoc review.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_TARGETS = ["bowl", "plate", "sake cup", "kamaboko", "corn", "onion segments"]
DEFAULT_MODEL = "/home/Groups/group2/.cache/modelscope/hub/models/Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_DATA_ROOT = "/home/Groups/group2/Working/tyy/data/lerf_ovs"
DEFAULT_LABEL_ROOT = "/home/Groups/group2/Working/tyy/data/lerf_ovs/label"
DEFAULT_ARTIFACT_DIR = "output/qwen_som_ramen_artifacts"
DEFAULT_OUT_PRED = "output/render_qwen_som_ramen/lerf"
DEFAULT_ORACLE_PRED = "output/render_qwen_som_ramen_oracle/lerf"
DEFAULT_SOURCES = [
    ("baseline", "output/render/lerf"),
    ("proposal_k1", "output/render_proposal_clip_cosine_k1/lerf"),
    ("proposal_k2", "output/render_proposal_clip_cosine_k2/lerf"),
    ("proposal_k3", "output/render_proposal_clip_cosine_k3/lerf"),
]
DEFAULT_DIAGNOSTIC_SOURCES = [
    ("scene_opt_diagnostic", "output/render_scene_opt/lerf"),
]
TARGET_AREA_PRIORS = {
    "bowl": (0.020, 0.105, 0.320),
    "plate": (0.018, 0.070, 0.180),
    "sake cup": (0.004, 0.022, 0.090),
    "kamaboko": (0.0004, 0.006, 0.035),
    "corn": (0.00015, 0.0017, 0.010),
    "onion segments": (0.00025, 0.006, 0.040),
}


@dataclass
class Candidate:
    candidate_id: int
    source: str
    kind: str
    mask: np.ndarray
    area_ratio: float
    score: float = 0.0
    path: str = ""
    level: int | None = None
    segment_id: int | None = None
    text_label: str = ""
    role: str = ""
    selectable: bool = True
    aliases: list[str] | None = None


def as_path(path: str | Path, root: Path | None = None) -> Path:
    p = Path(path)
    if p.is_absolute() or root is None:
        return p
    return root / p


def safe_name(prompt: str) -> str:
    return prompt.replace(" ", "_")


def norm_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def polygon_to_mask(shape: tuple[int, int], points_list: list[list[float]]) -> np.ndarray:
    points = np.asarray(points_list, dtype=np.int32)
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    return mask > 0


def gt_mask_from_annotation(annotation: dict[str, Any], prompt: str, shape: tuple[int, int]) -> np.ndarray:
    gt = np.zeros(shape, dtype=bool)
    for obj in annotation.get("objects", []):
        if obj.get("category") == prompt:
            gt |= polygon_to_mask(shape, obj["segmentation"])
    return gt


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_binary_mask(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray | None:
    if not path.exists():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if shape is not None and mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 128


def save_binary_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask.astype(np.uint8) * 255))


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / (union + 1e-6))


def mask_precision_recall(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    inter = np.logical_and(a, b).sum()
    pred = a.sum()
    gt = b.sum()
    return float(inter / (pred + 1e-6)), float(inter / (gt + 1e-6))


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0


def overlay_image(
    image: np.ndarray,
    mask: np.ndarray | None,
    color: tuple[int, int, int],
    alpha: float = 0.45,
) -> np.ndarray:
    out = image.copy().astype(np.float32)
    if mask is not None and mask.any():
        color_arr = np.asarray(color, dtype=np.float32)[None, None, :]
        out[mask] = (1.0 - alpha) * out[mask] + alpha * color_arr
    return np.clip(out, 0, 255).astype(np.uint8)


def font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def color_for_id(candidate_id: int) -> tuple[int, int, int]:
    palette = [
        (230, 57, 70),
        (29, 117, 243),
        (28, 160, 92),
        (244, 162, 97),
        (145, 78, 190),
        (44, 181, 202),
        (233, 196, 106),
        (214, 64, 159),
        (66, 133, 244),
        (77, 182, 172),
        (239, 83, 80),
        (126, 87, 194),
    ]
    return palette[(candidate_id - 1) % len(palette)]


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill=(255, 255, 255)) -> None:
    label_font = font(22)
    x, y = xy
    box = draw.textbbox((x, y), text, font=label_font)
    pad = 5
    draw.rectangle((box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad), fill=(0, 0, 0))
    draw.text((x, y), text, font=label_font, fill=fill)


def make_tile(
    image: np.ndarray,
    mask: np.ndarray | None,
    title: str,
    subtitle: str,
    candidate_id: int,
    tile_size: int,
) -> Image.Image:
    h, w = image.shape[:2]
    resized = cv2.resize(image, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
    color = color_for_id(max(candidate_id, 1))
    mask_resized = resize_mask(mask, (tile_size, tile_size)) if mask is not None else None
    tile_arr = overlay_image(resized, mask_resized, color, alpha=0.45)
    tile = Image.fromarray(tile_arr)
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, tile_size - 1, tile_size - 1), outline=(35, 35, 35), width=2)

    if mask is not None and mask.any():
        bbox = mask_bbox(mask)
        assert bbox is not None
        x0, y0, x1, y1 = bbox
        sx = tile_size / w
        sy = tile_size / h
        draw.rectangle(
            (int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)),
            outline=(255, 255, 255),
            width=3,
        )

        pad = max(12, int(max(x1 - x0, y1 - y0) * 0.75))
        cx0 = max(0, x0 - pad)
        cy0 = max(0, y0 - pad)
        cx1 = min(w, x1 + pad)
        cy1 = min(h, y1 + pad)
        if cx1 > cx0 and cy1 > cy0:
            crop_img = image[cy0:cy1, cx0:cx1]
            crop_mask = mask[cy0:cy1, cx0:cx1]
            inset_size = tile_size // 3
            crop_resized = cv2.resize(crop_img, (inset_size, inset_size), interpolation=cv2.INTER_AREA)
            crop_mask_resized = resize_mask(crop_mask, (inset_size, inset_size))
            crop_overlay = overlay_image(crop_resized, crop_mask_resized, color, alpha=0.55)
            inset = Image.fromarray(crop_overlay)
            ix = tile_size - inset_size - 8
            iy = tile_size - inset_size - 8
            tile.paste(inset, (ix, iy))
            draw.rectangle((ix, iy, ix + inset_size, iy + inset_size), outline=(255, 255, 255), width=2)

    draw_label(draw, (8, 8), title)
    if subtitle:
        sub_font = font(15)
        box = draw.textbbox((8, tile_size - 28), subtitle, font=sub_font)
        draw.rectangle((4, box[1] - 4, min(tile_size - 4, box[2] + 4), box[3] + 4), fill=(0, 0, 0))
        draw.text((8, tile_size - 28), subtitle[:42], font=sub_font, fill=(255, 255, 255))
    return tile


def make_som_panel(
    image: np.ndarray,
    candidates: list[Candidate],
    prompt: str,
    scene: str,
    frame: str,
    out_path: Path,
    tile_size: int = 360,
    cols: int = 3,
) -> None:
    visible = [c for c in candidates if c.selectable]
    tiles = [make_tile(image, None, "image", "candidate 0 = none", 0, tile_size)]
    for c in visible:
        title = f"#{c.candidate_id}"
        subtitle = f"{c.source} area={c.area_ratio:.3f}"
        if c.kind == "segment":
            subtitle = f"{c.source} L{c.level} S{c.segment_id} area={c.area_ratio:.3f}"
        tiles.append(make_tile(image, c.mask, title, subtitle, c.candidate_id, tile_size))

    rows = math.ceil(len(tiles) / cols)
    header_h = 70
    panel = Image.new("RGB", (cols * tile_size, rows * tile_size + header_h), (245, 245, 245))
    draw = ImageDraw.Draw(panel)
    header = f"SoM candidates | {scene}/{frame} | target: {prompt}"
    draw.text((14, 10), header, font=font(24), fill=(0, 0, 0))
    draw.text(
        (14, 42),
        "Qwen should return selected_ids. No GT is shown in this panel.",
        font=font(17),
        fill=(55, 55, 55),
    )
    for idx, tile in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        panel.paste(tile, (c * tile_size, header_h + r * tile_size))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path)


def make_comparison_panel(
    image: np.ndarray,
    prompt: str,
    frame: str,
    gt: np.ndarray,
    baseline: np.ndarray,
    selected: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    items = [
        ("image", None, (80, 80, 80)),
        ("GT", gt, (28, 160, 92)),
        ("baseline", baseline, (230, 57, 70)),
        (title, selected, (29, 117, 243)),
    ]
    tile_size = 360
    panel = Image.new("RGB", (len(items) * tile_size, tile_size + 55), (245, 245, 245))
    draw = ImageDraw.Draw(panel)
    draw.text((12, 8), f"{frame} | {prompt}", font=font(22), fill=(0, 0, 0))
    for i, (name, mask, color) in enumerate(items):
        tile = make_tile(image, mask, name, "", i + 1, tile_size)
        panel.paste(tile, (i * tile_size, 55))
        if mask is not None:
            iou = mask_iou(mask, gt)
            p, r = mask_precision_recall(mask, gt)
            draw.text(
                (i * tile_size + 10, 32),
                f"IoU={iou:.3f} P={p:.3f} R={r:.3f}",
                font=font(15),
                fill=color,
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path)


def candidate_to_record(c: Candidate, gt: np.ndarray | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "candidate_id": c.candidate_id,
        "source": c.source,
        "kind": c.kind,
        "area_ratio": c.area_ratio,
        "score": c.score,
        "path": c.path,
        "level": c.level,
        "segment_id": c.segment_id,
        "text_label": c.text_label,
        "role": c.role,
        "selectable": c.selectable,
        "aliases": c.aliases or [],
    }
    if gt is not None:
        row["gt_iou"] = mask_iou(c.mask, gt)
        p, r = mask_precision_recall(c.mask, gt)
        row["gt_precision"] = p
        row["gt_recall"] = r
    return row


def semantic_priority(target: str, label: str) -> int:
    t = norm_text(target)
    l = norm_text(label)
    if not t or not l:
        return 0
    if l == t:
        return 5
    if t in l or l in t:
        return 4
    t_words = set(t.split())
    l_words = set(l.split())
    if not t_words or not l_words:
        return 0
    if t_words & l_words:
        if "cup" in t_words and "cup" in l_words:
            return 3
        if list(t_words)[-1:] == list(l_words)[-1:]:
            return 2
        return 1
    return 0


def target_area_prior(target: str) -> tuple[float, float, float]:
    return TARGET_AREA_PRIORS.get(target, (0.0002, 0.020, 0.300))


def area_prior_score(target: str, area_ratio: float) -> float:
    min_area, center, max_area = target_area_prior(target)
    if area_ratio < min_area or area_ratio > max_area:
        return 0.0
    return float(math.exp(-0.75 * abs(math.log((area_ratio + 1e-9) / center))))


def deduplicate_candidates(candidates: list[Candidate], threshold: float) -> list[Candidate]:
    kept: list[Candidate] = []
    for cand in candidates:
        duplicate = None
        for existing in kept:
            if mask_iou(cand.mask, existing.mask) >= threshold:
                duplicate = existing
                break
        if duplicate is not None:
            alias = f"{cand.source}:{cand.kind}"
            if cand.kind == "segment":
                alias += f":L{cand.level}:S{cand.segment_id}"
            duplicate.aliases = duplicate.aliases or []
            duplicate.aliases.append(alias)
            continue
        cand.candidate_id = len(kept) + 1
        kept.append(cand)
    return kept


def add_prediction_candidates(
    candidates: list[Candidate],
    project_root: Path,
    sources: list[tuple[str, str]],
    scene: str,
    frame: str,
    prompt: str,
    shape: tuple[int, int],
    min_area: float,
    max_area: float,
) -> None:
    stem = safe_name(prompt) + ".png"
    for source_name, source_root in sources:
        path = as_path(source_root, project_root) / scene / frame / stem
        mask = load_binary_mask(path, shape)
        if mask is None or not mask.any():
            continue
        area = float(mask.mean())
        if area < min_area or area > max_area:
            continue
        candidates.append(
            Candidate(
                candidate_id=0,
                source=source_name,
                kind="prediction",
                mask=mask,
                area_ratio=area,
                score=0.0,
                path=str(path),
                selectable=True,
            )
        )


def add_segment_candidates(
    candidates: list[Candidate],
    data_root: Path,
    scene: str,
    frame: str,
    prompt: str,
    feature_dir: str,
    shape: tuple[int, int],
    min_area: float,
    max_area: float,
    max_segment_candidates: int,
) -> None:
    label_path = data_root / scene / feature_dir / f"{frame}_s.npy"
    meta_path = data_root / scene / feature_dir / f"{frame}_meta.json"
    if not label_path.exists() or not meta_path.exists():
        return
    label_maps = np.load(label_path, allow_pickle=True)
    meta = read_json(meta_path)
    meta_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for seg in meta.get("segments", []):
        meta_by_key[(int(seg.get("level", 0)), int(seg.get("segment_id", -1)))] = seg

    semantic_rows: list[tuple[float, Candidate]] = []
    generic_rows: list[tuple[float, Candidate]] = []
    for level in range(label_maps.shape[0]):
        labels = label_maps[level]
        for sid in np.unique(labels):
            sid_int = int(sid)
            if sid_int < 0:
                continue
            seg = meta_by_key.get((level, sid_int), {})
            label = str(seg.get("text_label", ""))
            priority = semantic_priority(prompt, label)
            mask = labels == sid_int
            if mask.shape != shape:
                mask = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            area = float(mask.mean())
            if area < min_area or area > max_area:
                continue
            text_score = float(seg.get("text_score", 0.0) or 0.0)
            role = str(seg.get("role", ""))
            role_bonus = 0.18 if role in {"object", "instance"} else 0.04
            level_bonus = 0.04 * level
            area_score = area_prior_score(prompt, area)
            if priority > 0:
                score = 2.0 * priority + role_bonus + level_bonus + text_score + 0.3 * area_score
                semantic_rows.append(
                    (
                        score,
                        Candidate(
                            candidate_id=0,
                            source="segment_map_semantic",
                            kind="segment",
                            mask=mask,
                            area_ratio=area,
                            score=score,
                            path=str(label_path),
                            level=level,
                            segment_id=sid_int,
                            text_label=label,
                            role=role,
                            selectable=True,
                        ),
                    )
                )
            if area_score > 0.0:
                score = area_score + role_bonus + level_bonus + 0.03 * text_score
                generic_rows.append(
                    (
                        score,
                        Candidate(
                            candidate_id=0,
                            source="segment_map_generic",
                            kind="segment",
                            mask=mask,
                            area_ratio=area,
                            score=score,
                            path=str(label_path),
                            level=level,
                            segment_id=sid_int,
                            text_label=label,
                            role=role,
                            selectable=True,
                        ),
                    )
                )

    semantic_rows.sort(key=lambda x: x[0], reverse=True)
    generic_rows.sort(key=lambda x: x[0], reverse=True)
    merged: list[Candidate] = []
    seen = set()
    semantic_take = max(4, max_segment_candidates // 2)
    generic_take = max_segment_candidates
    for _, cand in semantic_rows[:semantic_take]:
        key = (cand.level, cand.segment_id)
        if key not in seen:
            merged.append(cand)
            seen.add(key)
    for _, cand in generic_rows[:generic_take]:
        key = (cand.level, cand.segment_id)
        if key not in seen:
            merged.append(cand)
            seen.add(key)
    per_level: dict[int, int] = {}
    for _, cand in generic_rows:
        level = int(cand.level or 0)
        if per_level.get(level, 0) >= 3:
            continue
        key = (cand.level, cand.segment_id)
        if key in seen:
            continue
        merged.append(cand)
        seen.add(key)
        per_level[level] = per_level.get(level, 0) + 1
    candidates.extend(merged[: max_segment_candidates + semantic_take + 8])


def parse_sources(values: list[str] | None, defaults: list[tuple[str, str]]) -> list[tuple[str, str]]:
    if not values:
        return defaults
    out = []
    for item in values:
        if "=" not in item:
            raise ValueError(f"Source must be name=path, got {item}")
        name, path = item.split("=", 1)
        out.append((name, path))
    return out


def extract_json(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


def normalize_selection(raw: dict[str, Any] | None, valid_ids: set[int]) -> dict[str, Any]:
    if raw is None:
        raw = {}
    ids: list[int] = []
    if "selected_ids" in raw and isinstance(raw["selected_ids"], list):
        for value in raw["selected_ids"]:
            try:
                ids.append(int(value))
            except Exception:
                pass
    elif "selected_id" in raw:
        try:
            ids.append(int(raw["selected_id"]))
        except Exception:
            pass
    ids = [i for i in ids if i in valid_ids and i != 0]
    dedup_ids = []
    for i in ids:
        if i not in dedup_ids:
            dedup_ids.append(i)
    try:
        confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))
    except Exception:
        confidence = 0.0
    return {
        "selected_ids": dedup_ids,
        "selected_id": dedup_ids[0] if dedup_ids else 0,
        "confidence": confidence,
        "reason": str(raw.get("reason", ""))[:800],
        "raw": raw,
    }


def load_qwen_model(model_path: str):
    import importlib.util

    import torch
    extra_transformers_path = os.environ.get(
        "THGS_TRANSFORMERS_PATH",
        "/home/Groups/group2/Working/seg/miniconda3/envs/thgs/lib/python3.10/site-packages",
    )
    if extra_transformers_path and extra_transformers_path not in os.sys.path:
        os.sys.path.append(extra_transformers_path)
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    has_accelerate = importlib.util.find_spec("accelerate") is not None
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    errors = []
    for class_name in ("AutoModelForImageTextToText", "Qwen3VLForConditionalGeneration", "AutoModelForVision2Seq"):
        try:
            module = __import__("transformers", fromlist=[class_name])
            cls = getattr(module, class_name)
            kwargs = {"trust_remote_code": True}
            if has_accelerate and torch.cuda.is_available():
                kwargs["device_map"] = "auto"
            try:
                model = cls.from_pretrained(model_path, dtype=dtype, **kwargs)
            except TypeError:
                model = cls.from_pretrained(model_path, torch_dtype=dtype, **kwargs)
            if not has_accelerate and torch.cuda.is_available():
                model = model.to("cuda")
            model.eval()
            return torch, processor, model
        except Exception as exc:
            errors.append(f"{class_name}: {exc}")
    raise RuntimeError("Failed to load Qwen3-VL model. " + " | ".join(errors[-3:]))


def first_device(model):
    try:
        return next(model.parameters()).device
    except Exception:
        return "cpu"


def run_qwen_on_panel(
    bundle,
    panel: Path,
    target: str,
    frame: str,
    candidates: list[dict[str, Any]],
    max_new_tokens: int,
) -> tuple[dict[str, Any] | None, str]:
    torch, processor, model = bundle
    image = Image.open(panel).convert("RGB")
    candidate_text = "\n".join(
        f"- {c['candidate_id']}: source={c['source']}, area={c['area_ratio']:.4f}"
        for c in candidates
        if c.get("selectable", True)
    )
    prompt = (
        f"You are selecting the best segmentation mask for target object '{target}' "
        f"in a ramen scene frame '{frame}'.\n"
        "The image is a numbered segment-overlaid SoM panel. Candidate 0 means no candidate matches.\n"
        "Ignore any automatic candidate labels, source names, or segment ids; they can be wrong. "
        "Use only visual evidence from the image and the highlighted mask.\n"
        "Choose candidate ids that visually correspond to the target object. Prefer one complete mask; "
        "select up to 3 ids only if the object is split across masks. Avoid masks that mostly cover "
        "background, occluders, the whole scene, or a different object. A good mask should tightly cover "
        "the target object rather than merely overlap nearby ramen contents.\n"
        f"Candidate list:\n{candidate_text}\n"
        "Return JSON only with this schema: "
        '{"selected_ids":[int], "confidence": float, "reason": "short reason"}.'
    )
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    try:
        inputs = inputs.to(first_device(model))
    except Exception:
        pass
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    input_len = inputs["input_ids"].shape[-1]
    decoded = processor.batch_decode(generated[:, input_len:], skip_special_tokens=True)[0]
    return extract_json(decoded), decoded[-4000:]


def build_command(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    data_root = as_path(args.data_root, project_root)
    label_root = as_path(args.label_root, project_root)
    artifact_dir = as_path(args.artifact_dir, project_root)
    sources = parse_sources(args.source, DEFAULT_SOURCES)
    diagnostic_sources = parse_sources(args.diagnostic_source, DEFAULT_DIAGNOSTIC_SOURCES)
    if args.scene_opt_selectable:
        sources = sources + diagnostic_sources
    targets = set(args.targets)

    tasks: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    panel_dir = artifact_dir / "som_panels"
    candidate_mask_dir = artifact_dir / "candidate_masks"

    for json_path in sorted((label_root / args.scene).glob("frame_*.json")):
        frame = json_path.stem
        jpg_path = label_root / args.scene / f"{frame}.jpg"
        if not jpg_path.exists():
            continue
        image = load_rgb(jpg_path)
        h, w = image.shape[:2]
        shape = (h, w)
        annotation = read_json(json_path)
        prompts = sorted(set(obj["category"] for obj in annotation.get("objects", [])))
        for prompt in prompts:
            if prompt not in targets:
                continue
            gt = gt_mask_from_annotation(annotation, prompt, shape)
            if not gt.any():
                continue
            baseline_path = project_root / "output/render/lerf" / args.scene / frame / f"{safe_name(prompt)}.png"
            baseline = load_binary_mask(baseline_path, shape)
            if baseline is None:
                baseline = np.zeros(shape, dtype=bool)
            baseline_iou = mask_iou(baseline, gt)
            if baseline_iou > args.low_iou_threshold and not args.include_all_targets:
                continue

            candidates: list[Candidate] = []
            add_prediction_candidates(
                candidates,
                project_root,
                sources,
                args.scene,
                frame,
                prompt,
                shape,
                args.min_area,
                args.max_area,
            )
            add_segment_candidates(
                candidates,
                data_root,
                args.scene,
                frame,
                prompt,
                args.feature_dir,
                shape,
                args.min_area,
                args.max_area,
                args.max_segment_candidates,
            )
            candidates = deduplicate_candidates(candidates, args.dedup_iou)
            candidates = candidates[: args.max_candidates]

            if not candidates:
                continue

            for cand in candidates:
                mask_path = candidate_mask_dir / args.scene / frame / safe_name(prompt) / f"cand_{cand.candidate_id:02d}.png"
                save_binary_mask(mask_path, cand.mask)
                cand.path = str(mask_path)

            panel_path = panel_dir / args.scene / frame / f"{safe_name(prompt)}_som.png"
            make_som_panel(image, candidates, prompt, args.scene, frame, panel_path, args.tile_size, args.panel_cols)

            diag_records = []
            for diag_name, diag_root in diagnostic_sources:
                diag_mask = load_binary_mask(
                    as_path(diag_root, project_root) / args.scene / frame / f"{safe_name(prompt)}.png",
                    shape,
                )
                if diag_mask is not None:
                    diag_records.append(
                        {
                            "source": diag_name,
                            "iou": mask_iou(diag_mask, gt),
                            "area_ratio": float(diag_mask.mean()),
                        }
                    )

            cand_records = [candidate_to_record(c, gt) for c in candidates]
            best = max(cand_records, key=lambda x: x["gt_iou"])
            task = {
                "scene": args.scene,
                "frame": frame,
                "target": prompt,
                "target_safe": safe_name(prompt),
                "image": str(jpg_path),
                "panel": str(panel_path),
                "baseline_iou": baseline_iou,
                "baseline_area_ratio": float(baseline.mean()),
                "gt_area_ratio": float(gt.mean()),
                "oracle_candidate_id": best["candidate_id"],
                "oracle_iou": best["gt_iou"],
                "diagnostic_sources": diag_records,
                "candidates": cand_records,
            }
            tasks.append(task)
            case_rows.append(
                {
                    "scene": args.scene,
                    "frame": frame,
                    "target": prompt,
                    "baseline_iou": baseline_iou,
                    "oracle_iou": best["gt_iou"],
                    "oracle_candidate_id": best["candidate_id"],
                    "num_candidates": len(candidates),
                    "panel": str(panel_path),
                }
            )

    write_jsonl(artifact_dir / "tasks.jsonl", tasks)
    write_jsonl(artifact_dir / "candidates.jsonl", case_rows)
    write_json(artifact_dir / "build_summary.json", summarize_cases(tasks))
    print(json.dumps({"tasks": len(tasks), "artifact_dir": str(artifact_dir)}, indent=2))
    return 0


def summarize_cases(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    by_target: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        by_target.setdefault(task["target"], []).append(task)
    target_summary = {}
    for target, rows in by_target.items():
        target_summary[target] = {
            "n": len(rows),
            "baseline_iou_mean": float(np.mean([r["baseline_iou"] for r in rows])),
            "oracle_iou_mean": float(np.mean([r["oracle_iou"] for r in rows])),
            "candidate_recall_at_0_25": float(np.mean([r["oracle_iou"] >= 0.25 for r in rows])),
            "candidate_recall_at_0_50": float(np.mean([r["oracle_iou"] >= 0.50 for r in rows])),
        }
    return {
        "num_tasks": len(tasks),
        "targets": target_summary,
        "note": "oracle_iou uses GT only for analysis; it is not shown in SoM panels or Qwen prompts.",
    }


def qwen_command(args: argparse.Namespace) -> int:
    artifact_dir = Path(args.artifact_dir).resolve()
    tasks = read_jsonl(artifact_dir / "tasks.jsonl")
    if args.max_items:
        tasks = tasks[: args.max_items]
    bundle = load_qwen_model(args.model)
    rows = []
    for idx, task in enumerate(tasks, 1):
        valid_ids = {0}
        valid_ids.update(int(c["candidate_id"]) for c in task["candidates"] if c.get("selectable", True))
        raw, raw_text = run_qwen_on_panel(
            bundle,
            Path(task["panel"]),
            task["target"],
            task["frame"],
            task["candidates"],
            args.max_new_tokens,
        )
        normalized = normalize_selection(raw, valid_ids)
        selected_iou = 0.0
        if normalized["selected_ids"]:
            selected = union_candidate_masks(task, normalized["selected_ids"])
            gt = load_gt_for_task(task)
            selected_iou = mask_iou(selected, gt)
        row = {
            "scene": task["scene"],
            "frame": task["frame"],
            "target": task["target"],
            "panel": task["panel"],
            "baseline_iou": task["baseline_iou"],
            "oracle_iou": task["oracle_iou"],
            "oracle_candidate_id": task["oracle_candidate_id"],
            "selected_iou": selected_iou,
            **normalized,
            "raw_text_tail": raw_text,
        }
        rows.append(row)
        print(json.dumps({"idx": idx, "n": len(tasks), "frame": task["frame"], "target": task["target"], "ids": row["selected_ids"], "iou": selected_iou}, ensure_ascii=False), flush=True)
    write_jsonl(artifact_dir / args.selection_file, rows)
    write_json(artifact_dir / "qwen_summary.json", summarize_selections(rows))
    return 0


def summarize_selections(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_target: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_target.setdefault(row["target"], []).append(row)
    out = {"num_rows": len(rows), "targets": {}}
    for target, target_rows in by_target.items():
        out["targets"][target] = {
            "n": len(target_rows),
            "baseline_iou_mean": float(np.mean([r["baseline_iou"] for r in target_rows])),
            "qwen_iou_mean": float(np.mean([r["selected_iou"] for r in target_rows])),
            "oracle_iou_mean": float(np.mean([r["oracle_iou"] for r in target_rows])),
            "selected_nonzero_rate": float(np.mean([bool(r["selected_ids"]) for r in target_rows])),
        }
    return out


def load_gt_for_task(task: dict[str, Any]) -> np.ndarray:
    image = load_rgb(Path(task["image"]))
    shape = image.shape[:2]
    label_root = Path(task["image"]).parent
    annotation = read_json(label_root / f"{task['frame']}.json")
    return gt_mask_from_annotation(annotation, task["target"], shape)


def union_candidate_masks(task: dict[str, Any], selected_ids: list[int]) -> np.ndarray:
    image = load_rgb(Path(task["image"]))
    out = np.zeros(image.shape[:2], dtype=bool)
    by_id = {int(c["candidate_id"]): c for c in task["candidates"]}
    for cid in selected_ids:
        cand = by_id.get(int(cid))
        if not cand:
            continue
        mask = load_binary_mask(Path(cand["path"]), out.shape)
        if mask is not None:
            out |= mask
    return out


def selection_rows_by_key(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = {}
    for row in read_jsonl(path):
        rows[(row["frame"], row["target"])] = row
    return rows


def apply_command(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    artifact_dir = as_path(args.artifact_dir, project_root)
    tasks = read_jsonl(artifact_dir / "tasks.jsonl")
    baseline_pred = as_path(args.baseline_pred, project_root)
    out_pred = as_path(args.out_pred, project_root)
    scene_src = baseline_pred / args.scene
    scene_dst = out_pred / args.scene
    if scene_dst.exists():
        shutil.rmtree(scene_dst)
    shutil.copytree(scene_src, scene_dst)

    selections = selection_rows_by_key(artifact_dir / args.selection_file)
    records = []
    for task in tasks:
        image = load_rgb(Path(task["image"]))
        gt = load_gt_for_task(task)
        baseline = load_binary_mask(baseline_pred / args.scene / task["frame"] / f"{task['target_safe']}.png", gt.shape)
        if baseline is None:
            baseline = np.zeros(gt.shape, dtype=bool)

        selected_ids: list[int] = []
        mode_name = args.mode
        if args.mode == "qwen":
            row = selections.get((task["frame"], task["target"]), {})
            selected_ids = [int(x) for x in row.get("selected_ids", [])]
        elif args.mode == "oracle":
            selected_ids = [int(task["oracle_candidate_id"])]
            mode_name = "oracle"
        else:
            raise ValueError(args.mode)

        selected = union_candidate_masks(task, selected_ids) if selected_ids else baseline.copy()
        out_mask = scene_dst / task["frame"] / f"{task['target_safe']}.png"
        save_binary_mask(out_mask, selected)

        compare_path = artifact_dir / "applied_compare" / args.mode / args.scene / task["frame"] / f"{task['target_safe']}.png"
        make_comparison_panel(image, task["target"], task["frame"], gt, baseline, selected, compare_path, mode_name)

        records.append(
            {
                "scene": task["scene"],
                "frame": task["frame"],
                "target": task["target"],
                "selected_ids": selected_ids,
                "baseline_iou": mask_iou(baseline, gt),
                "applied_iou": mask_iou(selected, gt),
                "oracle_iou": task["oracle_iou"],
                "out_mask": str(out_mask),
                "compare": str(compare_path),
            }
        )
    write_jsonl(artifact_dir / f"applied_{args.mode}.jsonl", records)
    write_report(artifact_dir / f"report_{args.mode}", records)
    print(json.dumps({"applied": len(records), "out_pred": str(out_pred), "mode": args.mode}, indent=2))
    return 0


def write_report(prefix: Path, rows: list[dict[str, Any]]) -> None:
    csv_path = prefix.with_suffix(".csv")
    md_path = prefix.with_suffix(".md")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["scene", "frame", "target", "selected_ids", "baseline_iou", "applied_iou", "oracle_iou", "compare"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})

    by_target: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_target.setdefault(row["target"], []).append(row)
    lines = [
        "| target | n | baseline IoU | applied IoU | oracle IoU | delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for target in sorted(by_target):
        target_rows = by_target[target]
        baseline = float(np.mean([r["baseline_iou"] for r in target_rows]))
        applied = float(np.mean([r["applied_iou"] for r in target_rows]))
        oracle = float(np.mean([r["oracle_iou"] for r in target_rows]))
        lines.append(f"| {target} | {len(target_rows)} | {baseline:.4f} | {applied:.4f} | {oracle:.4f} | {applied - baseline:+.4f} |")
    if rows:
        baseline = float(np.mean([r["baseline_iou"] for r in rows]))
        applied = float(np.mean([r["applied_iou"] for r in rows]))
        oracle = float(np.mean([r["oracle_iou"] for r in rows]))
        lines.extend(
            [
                "",
                f"Overall selected cases: n={len(rows)}, baseline={baseline:.4f}, applied={applied:.4f}, oracle={oracle:.4f}, delta={applied - baseline:+.4f}",
            ]
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def report_command(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    artifact_dir = as_path(args.artifact_dir, project_root)
    rows = read_jsonl(artifact_dir / f"applied_{args.mode}.jsonl")
    write_report(artifact_dir / f"report_{args.mode}", rows)
    print((artifact_dir / f"report_{args.mode}.md").read_text(encoding="utf-8"))
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--project_root", default=".")
    common.add_argument("--artifact_dir", default=DEFAULT_ARTIFACT_DIR)
    common.add_argument("--scene", default="ramen")

    build = sub.add_parser("build", parents=[common])
    build.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    build.add_argument("--label_root", default=DEFAULT_LABEL_ROOT)
    build.add_argument("--feature_dir", default="language_features_part_only_update")
    build.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    build.add_argument("--source", action="append", help="Selectable source as name=path. May repeat.")
    build.add_argument("--diagnostic_source", action="append", help="Diagnostic source as name=path. May repeat.")
    build.add_argument("--scene_opt_selectable", action="store_true")
    build.add_argument("--low_iou_threshold", type=float, default=0.10)
    build.add_argument("--include_all_targets", action="store_true")
    build.add_argument("--min_area", type=float, default=0.00002)
    build.add_argument("--max_area", type=float, default=0.75)
    build.add_argument("--max_segment_candidates", type=int, default=8)
    build.add_argument("--max_candidates", type=int, default=14)
    build.add_argument("--dedup_iou", type=float, default=0.98)
    build.add_argument("--tile_size", type=int, default=340)
    build.add_argument("--panel_cols", type=int, default=3)
    build.set_defaults(func=build_command)

    qwen = sub.add_parser("qwen", parents=[common])
    qwen.add_argument("--model", default=DEFAULT_MODEL)
    qwen.add_argument("--selection_file", default="selections_qwen.jsonl")
    qwen.add_argument("--max_new_tokens", type=int, default=192)
    qwen.add_argument("--max_items", type=int, default=0)
    qwen.set_defaults(func=qwen_command)

    apply = sub.add_parser("apply", parents=[common])
    apply.add_argument("--baseline_pred", default="output/render/lerf")
    apply.add_argument("--out_pred", default=DEFAULT_OUT_PRED)
    apply.add_argument("--selection_file", default="selections_qwen.jsonl")
    apply.add_argument("--mode", choices=["qwen", "oracle"], default="qwen")
    apply.set_defaults(func=apply_command)

    report = sub.add_parser("report", parents=[common])
    report.add_argument("--mode", choices=["qwen", "oracle"], default="qwen")
    report.set_defaults(func=report_command)
    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
