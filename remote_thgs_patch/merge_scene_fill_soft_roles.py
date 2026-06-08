import argparse
import glob
import json
import os
import shutil


import cv2
import numpy as np

from query_reasoner import heuristic_query_plan


def safe_name(prompt):
    return prompt.replace(" ", "_")


def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return mask > 128


def mask_iou(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter / (union + 1e-6))


def should_replace(base_path, soft_path, role, args):
    if not args.guard_replacement:
        return True, "unguarded"
    if role == "modifier" and args.always_replace_modifier:
        return True, "modifier"

    base = load_mask(base_path)
    soft = load_mask(soft_path)
    if soft is None:
        return False, "missing_soft"
    if base is None:
        return True, "missing_base"
    if base.shape != soft.shape:
        soft = cv2.resize(soft.astype(np.uint8), (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST) > 0

    base_area = float(base.mean())
    soft_area = float(soft.mean())
    if soft_area <= 0:
        return False, "empty_soft"
    if base_area <= args.base_empty_area:
        return True, "base_empty"

    overlap = mask_iou(base, soft)
    area_ratio = soft_area / max(base_area, 1e-8)
    if overlap <= args.max_overlap_iou and area_ratio <= args.max_area_ratio:
        return True, "low_overlap"
    if args.allow_shrink_replace and soft_area <= base_area * args.shrink_area_ratio:
        return True, "shrink"
    return False, f"guarded:overlap={overlap:.4f},area_ratio={area_ratio:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_root", default="/home/Groups/group2/Working/tyy/data/lerf_ovs/label")
    parser.add_argument("--base_root", default="output/render_scene_opt/lerf")
    parser.add_argument("--soft_root", default="output/render_hier_soft_roles_l23/lerf")
    parser.add_argument("--out_root", default="output/render_scene_fill_soft_roles_l23/lerf")
    parser.add_argument("--roles", nargs="+", default=["part", "modifier"], choices=["object", "part", "modifier"])
    parser.add_argument("--scenes", nargs="+", default=["figurines", "ramen", "teatime", "waldo_kitchen"])
    parser.add_argument("--replace_scenes", nargs="+", default=None)
    parser.add_argument("--guard_replacement", action="store_true")
    parser.add_argument("--base_empty_area", type=float, default=0.0015)
    parser.add_argument("--max_overlap_iou", type=float, default=0.12)
    parser.add_argument("--max_area_ratio", type=float, default=3.0)
    parser.add_argument("--allow_shrink_replace", action="store_true")
    parser.add_argument("--shrink_area_ratio", type=float, default=1.05)
    parser.add_argument("--always_replace_modifier", action="store_true")
    args = parser.parse_args()

    roles = set(args.roles)
    replace_scenes = set(args.replace_scenes or args.scenes)
    replaced = []
    for scene in args.scenes:
        scene_label_dir = os.path.join(args.label_root, scene)
        for anno_path in sorted(glob.glob(os.path.join(scene_label_dir, "frame_*.json"))):
            frame = os.path.splitext(os.path.basename(anno_path))[0]
            base_dir = os.path.join(args.base_root, scene, frame)
            soft_dir = os.path.join(args.soft_root, scene, frame)
            out_dir = os.path.join(args.out_root, scene, frame)
            if not os.path.isdir(base_dir):
                continue
            os.makedirs(out_dir, exist_ok=True)
            for src in glob.glob(os.path.join(base_dir, "*.png")):
                shutil.copyfile(src, os.path.join(out_dir, os.path.basename(src)))

            with open(anno_path, "r", encoding="utf-8") as f:
                anno = json.load(f)
            prompts = sorted(set(obj["category"] for obj in anno["objects"]))
            for prompt in prompts:
                if scene not in replace_scenes:
                    continue
                plan = heuristic_query_plan(prompt, prompts)
                if plan.role not in roles:
                    continue
                name = safe_name(prompt) + ".png"
                soft_path = os.path.join(soft_dir, name)
                base_path = os.path.join(base_dir, name)
                if not os.path.exists(soft_path):
                    continue
                ok, reason = should_replace(base_path, soft_path, plan.role, args)
                if not ok:
                    continue
                shutil.copyfile(soft_path, os.path.join(out_dir, name))
                replaced.append([scene, frame, prompt, plan.role, reason])

    print("replaced", len(replaced))
    for item in replaced[:80]:
        print(*item)


if __name__ == "__main__":
    main()
