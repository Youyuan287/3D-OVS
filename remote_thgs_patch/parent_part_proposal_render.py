import argparse
import glob
import json
import math
import os
import shutil

import cv2
import numpy as np

from query_reasoner import heuristic_query_plan


def safe_name(prompt):
    return prompt.replace(" ", "_")


def part_head(prompt):
    tokens = prompt.replace("-", " ").split()
    return tokens[-1].lower() if tokens else ""


def target_cover_for_prompt(prompt, args):
    head = part_head(prompt)
    if head in {"hooves", "hoof"}:
        return args.hoof_target_cover
    if head == "nose":
        return args.nose_target_cover
    if head in {"hand", "handle", "hat"}:
        return args.small_part_target_cover
    return args.target_cover


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


def is_part_prompt(prompt, labels):
    return heuristic_query_plan(prompt, labels).role == "part"


def load_mask(path, shape=None):
    if not os.path.exists(path):
        return None
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if shape is not None and mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 128


def mask_iou(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter / (union + 1e-6))


def load_anchor(base_dir, anchors, shape, dilate):
    out = np.zeros(shape, dtype=bool)
    for anchor in anchors:
        mask = load_mask(os.path.join(base_dir, safe_name(anchor) + ".png"), shape)
        if mask is not None:
            out |= mask
    if out.any() and dilate > 0:
        out = cv2.dilate(out.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=dilate) > 0
    return out if out.any() else None


def choose_part_mask(label_maps, meta, prompt, labels, anchor_mask, args):
    plan = heuristic_query_plan(prompt, labels)
    positives = set([prompt])
    positives.update(plan.target_terms or [])
    positives.update(plan.part_terms or [])
    positive_heads = {p.split()[-1] for p in positives if p}
    h, w = anchor_mask.shape
    image_area = h * w
    anchor_area = max(float(anchor_mask.sum()), 1.0)

    by_key = {(int(s.get("level", 0)), int(s.get("segment_id", -1))): s for s in meta.get("segments", [])}
    ranked = []
    for layer in range(label_maps.shape[0]):
        labels_map = label_maps[layer]
        for sid in np.unique(labels_map):
            if sid < 0:
                continue
            mask = labels_map == sid
            area = float(mask.sum())
            if area <= 0:
                continue
            area_ratio = area / image_area
            if area_ratio < args.min_area or area_ratio > args.max_area:
                continue
            inter = float(np.logical_and(mask, anchor_mask).sum())
            containment = inter / area
            if containment < args.min_containment:
                continue
            cover = inter / anchor_area
            seg = by_key.get((layer, int(sid)), {})
            text_label = seg.get("text_label", "")
            text_head = text_label.split()[-1] if text_label else ""
            semantic_bonus = 0.0
            if text_label in positives:
                semantic_bonus += args.exact_bonus
            elif text_head in positive_heads:
                semantic_bonus += args.head_bonus
            if seg.get("role") == "part":
                semantic_bonus += args.part_role_bonus

            target_cover = target_cover_for_prompt(prompt, args)
            cover_penalty = abs(math.log((cover + 1e-6) / target_cover))
            same_label_area_penalty = 0.0
            if text_label in positives and cover > args.same_label_max_cover:
                same_label_area_penalty = args.same_label_area_penalty * (
                    cover / max(args.same_label_max_cover, 1e-6) - 1.0
                )
            score = (
                args.containment_weight * containment
                - args.cover_penalty * cover_penalty
                - args.level_penalty * max(layer - 2, 0)
                + semantic_bonus
                - same_label_area_penalty
            )
            ranked.append((score, mask, {
                "layer": int(layer),
                "segment_id": int(sid),
                "area": area,
                "area_ratio": area_ratio,
                "cover": cover,
                "containment": containment,
                "text_label": text_label,
                "same_label_area_penalty": same_label_area_penalty,
                "score": score,
            }))

    if args.consistency_bonus > 0 and ranked:
        adjusted = []
        for score, mask, info in ranked:
            support = 0.0
            area = max(float(info["area"]), 1.0)
            for _, other_mask, other_info in ranked:
                if other_info is info:
                    continue
                if abs(other_info["layer"] - info["layer"]) > args.consistency_max_level_gap:
                    continue
                other_area = max(float(other_info["area"]), 1.0)
                area_ratio = max(area, other_area) / min(area, other_area)
                if area_ratio > args.consistency_max_area_ratio:
                    continue
                support = max(support, mask_iou(mask, other_mask))
            consistency_bonus = args.consistency_bonus * support
            info = dict(info)
            info["cross_layer_support"] = support
            info["consistency_bonus"] = consistency_bonus
            info["score"] = float(score + consistency_bonus)
            adjusted.append((score + consistency_bonus, mask, info))
        ranked = adjusted
    ranked.sort(key=lambda x: x[0], reverse=True)
    if not ranked:
        return None, []
    selected = ranked[:args.select_topk]
    out = np.zeros_like(anchor_mask)
    debug = []
    for _, mask, info in selected:
        out |= mask
        debug.append(info)
    out &= anchor_mask
    return out, debug


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/home/Groups/group2/Working/tyy/data/lerf_ovs")
    parser.add_argument("--label_root", default="/home/Groups/group2/Working/tyy/data/lerf_ovs/label")
    parser.add_argument("--baseline_pred", default="/home/Groups/group2/Working/tyy/project/THGS-main/output/render/lerf")
    parser.add_argument("--out_root", default="/home/Groups/group2/Working/tyy/project/THGS-main/output/render_parent_part/lerf")
    parser.add_argument("--feature_dir", default="language_features_part_only_update")
    parser.add_argument("--scenes", nargs="+", default=["figurines", "ramen", "teatime", "waldo_kitchen"])
    parser.add_argument("--anchor_dilate", type=int, default=1)
    parser.add_argument("--min_area", type=float, default=0.00002)
    parser.add_argument("--max_area", type=float, default=0.12)
    parser.add_argument("--min_containment", type=float, default=0.55)
    parser.add_argument("--target_cover", type=float, default=0.10)
    parser.add_argument("--nose_target_cover", type=float, default=0.13)
    parser.add_argument("--hoof_target_cover", type=float, default=0.025)
    parser.add_argument("--small_part_target_cover", type=float, default=0.04)
    parser.add_argument("--containment_weight", type=float, default=2.0)
    parser.add_argument("--cover_penalty", type=float, default=0.18)
    parser.add_argument("--level_penalty", type=float, default=0.05)
    parser.add_argument("--exact_bonus", type=float, default=0.12)
    parser.add_argument("--head_bonus", type=float, default=0.1)
    parser.add_argument("--part_role_bonus", type=float, default=0.05)
    parser.add_argument("--same_label_max_cover", type=float, default=0.18)
    parser.add_argument("--same_label_area_penalty", type=float, default=0.35)
    parser.add_argument("--consistency_bonus", type=float, default=0.12)
    parser.add_argument("--consistency_max_level_gap", type=int, default=1)
    parser.add_argument("--consistency_max_area_ratio", type=float, default=1.35)
    parser.add_argument("--select_topk", type=int, default=1)
    args = parser.parse_args()

    debug = {}
    for scene in args.scenes:
        debug[scene] = {}
        for frame_dir in sorted(glob.glob(os.path.join(args.baseline_pred, scene, "frame_*"))):
            frame = os.path.basename(frame_dir)
            out_dir = os.path.join(args.out_root, scene, frame)
            os.makedirs(out_dir, exist_ok=True)
            for src in glob.glob(os.path.join(frame_dir, "*.png")):
                shutil.copyfile(src, os.path.join(out_dir, os.path.basename(src)))

            anno_path = os.path.join(args.label_root, scene, frame + ".json")
            if not os.path.exists(anno_path):
                continue
            anno = json.load(open(anno_path))
            prompts = sorted(set(obj["category"] for obj in anno["objects"]))
            label_path = os.path.join(args.data_root, scene, args.feature_dir, frame + "_s.npy")
            meta_path = os.path.join(args.data_root, scene, args.feature_dir, frame + "_meta.json")
            if not os.path.exists(label_path) or not os.path.exists(meta_path):
                continue
            label_maps = np.load(label_path, allow_pickle=True)
            meta = json.load(open(meta_path))
            shape = label_maps.shape[1:]

            for prompt in prompts:
                if not is_part_prompt(prompt, prompts):
                    continue
                anchors = anchor_candidates(prompt, prompts)
                anchor_mask = load_anchor(frame_dir, anchors, shape, args.anchor_dilate)
                if anchor_mask is None:
                    continue
                mask, selected = choose_part_mask(label_maps, meta, prompt, prompts, anchor_mask, args)
                if mask is None:
                    continue
                cv2.imwrite(os.path.join(out_dir, safe_name(prompt) + ".png"), mask.astype(np.uint8) * 255)
                debug.setdefault(scene, {}).setdefault(frame, {})[prompt] = {
                    "anchors": anchors,
                    "selected": selected,
                }

    os.makedirs(os.path.dirname(args.out_root), exist_ok=True)
    json.dump(debug, open(os.path.join(os.path.dirname(args.out_root), "parent_part_debug.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
