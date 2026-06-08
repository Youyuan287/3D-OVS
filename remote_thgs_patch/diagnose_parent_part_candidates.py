import json
import math
import os

import cv2
import numpy as np

from query_reasoner import heuristic_query_plan


DATA_ROOT = "/home/Groups/group2/Working/tyy/data/lerf_ovs"
BASE_ROOT = "output/render/lerf"
FEATURE_DIR = "language_features_part_only_update"

CASES = [
    ("teatime", "frame_00002", "bear nose"),
    ("teatime", "frame_00025", "bear nose"),
    ("teatime", "frame_00025", "hooves"),
    ("teatime", "frame_00043", "hooves"),
    ("teatime", "frame_00107", "bear nose"),
]

PARAMS = {
    "min_area": 0.00002,
    "max_area": 0.12,
    "min_containment": 0.55,
    "target_cover": 0.08,
    "containment_weight": 2.0,
    "cover_penalty": 0.18,
    "level_penalty": 0.05,
    "exact_bonus": 0.25,
    "head_bonus": 0.1,
    "part_role_bonus": 0.05,
}


def safe_name(prompt):
    return prompt.replace(" ", "_")


def load_mask(path, shape=None):
    if not os.path.exists(path):
        return None
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if shape is not None and mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 128


def iou(a, b):
    return float(np.logical_and(a, b).sum() / (np.logical_or(a, b).sum() + 1e-6))


def anchor_candidates(prompt, labels):
    plan = heuristic_query_plan(prompt, labels)
    candidates = []
    for term in plan.anchor_terms or []:
        candidates.extend([label for label in labels if label != prompt and term and term in label])
    if prompt == "hooves":
        candidates.extend([label for label in labels if label in {"sheep", "stuffed bear", "porcelain hand"}])
    if prompt == "hand":
        candidates.extend([label for label in labels if label in {"porcelain hand"}])
    out = []
    for label in candidates:
        if label not in out:
            out.append(label)
    return out


def score_candidate(mask, layer, sid, meta_by_key, prompt, labels, anchor_mask):
    plan = heuristic_query_plan(prompt, labels)
    positives = set([prompt])
    positives.update(plan.target_terms or [])
    positives.update(plan.part_terms or [])
    positive_heads = {p.split()[-1] for p in positives if p}

    h, w = anchor_mask.shape
    image_area = h * w
    anchor_area = max(float(anchor_mask.sum()), 1.0)
    area = float(mask.sum())
    area_ratio = area / image_area
    if area_ratio < PARAMS["min_area"] or area_ratio > PARAMS["max_area"]:
        return None

    inter = float(np.logical_and(mask, anchor_mask).sum())
    containment = inter / (area + 1e-6)
    if containment < PARAMS["min_containment"]:
        return None
    cover = inter / anchor_area
    seg = meta_by_key.get((layer, int(sid)), {})
    text_label = seg.get("text_label", "")
    text_head = text_label.split()[-1] if text_label else ""
    semantic_bonus = 0.0
    if text_label in positives:
        semantic_bonus += PARAMS["exact_bonus"]
    elif text_head in positive_heads:
        semantic_bonus += PARAMS["head_bonus"]
    if seg.get("role") == "part":
        semantic_bonus += PARAMS["part_role_bonus"]

    cover_penalty = abs(math.log((cover + 1e-6) / PARAMS["target_cover"]))
    score = (
        PARAMS["containment_weight"] * containment
        - PARAMS["cover_penalty"] * cover_penalty
        - PARAMS["level_penalty"] * max(layer - 2, 0)
        + semantic_bonus
    )
    return {
        "layer": int(layer),
        "segment_id": int(sid),
        "area": area,
        "area_ratio": area_ratio,
        "cover": cover,
        "containment": containment,
        "text_label": text_label,
        "role": seg.get("role"),
        "score": float(score),
    }


def main():
    for scene, frame, prompt in CASES:
        anno_path = os.path.join(DATA_ROOT, "label", scene, frame + ".json")
        anno = json.load(open(anno_path))
        labels = sorted(set(obj["category"] for obj in anno["objects"]))
        label_maps = np.load(os.path.join(DATA_ROOT, scene, FEATURE_DIR, frame + "_s.npy"), allow_pickle=True)
        meta = json.load(open(os.path.join(DATA_ROOT, scene, FEATURE_DIR, frame + "_meta.json")))
        meta_by_key = {
            (int(s.get("level", 0)), int(s.get("segment_id", -1))): s
            for s in meta.get("segments", [])
        }
        shape = label_maps.shape[1:]

        anchor_mask = np.zeros(shape, dtype=bool)
        anchors = anchor_candidates(prompt, labels)
        for anchor in anchors:
            mask = load_mask(os.path.join(BASE_ROOT, scene, frame, safe_name(anchor) + ".png"), shape)
            if mask is not None:
                anchor_mask |= mask
        if anchor_mask.any():
            anchor_mask = cv2.dilate(anchor_mask.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=1) > 0

        gt = load_mask(os.path.join(BASE_ROOT, scene, frame, safe_name(prompt) + "_gt.png"), shape)
        if gt is None or not anchor_mask.any():
            print("SKIP", scene, frame, prompt)
            continue

        rows = []
        for layer in range(label_maps.shape[0]):
            labels_map = label_maps[layer]
            for sid in np.unique(labels_map):
                if sid < 0:
                    continue
                mask = labels_map == sid
                scored = score_candidate(mask, layer, sid, meta_by_key, prompt, labels, anchor_mask)
                if scored is None:
                    continue
                scored["gt_iou"] = iou(mask, gt)
                rows.append(scored)

        print(f"\nCASE {scene} {frame} {prompt} anchors={anchors} gt_area={int(gt.sum())} candidates={len(rows)}")
        print("top_score")
        for row in sorted(rows, key=lambda x: x["score"], reverse=True)[:8]:
            print(json.dumps(row, ensure_ascii=False, sort_keys=True))
        print("top_gt_iou")
        for row in sorted(rows, key=lambda x: x["gt_iou"], reverse=True)[:8]:
            print(json.dumps(row, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
