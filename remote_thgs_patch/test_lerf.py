import os
import torch
from random import randint
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
import cv2
import json
import numpy as np
import torch.nn.functional as F
from utils.vlm_utils import ClipSimMeasure
from nag_data import SemanticNAG
from query_reasoner import QueryReasoner
from proposal_reasoner import ProposalReasoner


PART_HINTS = {
    "nose", "hooves", "hoof", "hand", "hat", "handle", "leg", "arm",
}

ROUTING_STOPWORDS = {"with", "of", "the", "a", "an", "on", "in", "and"}


def routing_terms(prompt):
    tokens = [
        token.strip(" ,.;:!?()[]{}\"'").lower()
        for token in prompt.replace("-", " ").split()
    ]
    tokens = [token for token in tokens if token and token not in ROUTING_STOPWORDS]
    terms = [prompt]
    terms.extend(tokens)
    terms.extend([" ".join(tokens[i:i + 2]) for i in range(len(tokens) - 1)])

    deduped = []
    for term in terms:
        if term and term not in deduped:
            deduped.append(term)
    return deduped


def prompt_parts(prompt):
    parts = [prompt]
    if " with " in prompt:
        tail = prompt.split(" with ", 1)[1].strip()
        if tail:
            parts.insert(0, tail)

    tokens = prompt.replace("-", " ").split()
    if tokens and tokens[-1].lower() in PART_HINTS:
        parts.insert(0, tokens[-1])

    deduped = []
    for part in parts:
        if part and part not in deduped:
            deduped.append(part)
    return deduped


def is_part_prompt(prompt):
    tokens = prompt.replace("-", " ").split()
    return bool(tokens and tokens[-1].lower() in PART_HINTS)


def is_modifier_prompt(prompt):
    return " with " in prompt


def should_soft_route(prompt, plan, args):
    if args.semantic_mode not in ["soft", "hier_soft"]:
        return False
    roles = set(args.soft_route_roles or [])
    return plan.role in roles


def combined_similarity(vlm, features, queries, weights):
    sim_accum = None
    total_weight = 0.0
    for query, weight in zip(queries, weights):
        vlm.encode_text(query)
        sim_list = [vlm.compute_similarity(f) for f in features]
        if sim_accum is None:
            sim_accum = [weight * sim for sim in sim_list]
        else:
            sim_accum = [acc + weight * sim for acc, sim in zip(sim_accum, sim_list)]
        total_weight += weight
    return [sim / max(total_weight, 1e-6) for sim in sim_accum]


def similarity_for_query(vlm, features, query):
    vlm.encode_text(query)
    return [vlm.compute_similarity(f) for f in features]


def similarity_for_terms(vlm, features, terms, weights=None):
    terms = [term for term in terms if term]
    if not terms:
        raise ValueError("At least one text term is required")
    if weights is None:
        weights = [1.0] * len(terms)
    return combined_similarity(vlm, features, terms, weights)


def soft_routed_gaussian(snag, vlm, features, prompt, args):
    terms = routing_terms(prompt)
    vlm.encode_text(prompt)
    sim_full = [vlm.compute_similarity(f) for f in features]
    sim_terms = []
    for term in terms:
        vlm.encode_text(term)
        sim_terms.append([vlm.compute_similarity(f) for f in features])

    point_scores, debug = snag.get_soft_gaussian_scores(
        sim_full,
        sim_terms,
        levels=args.routing_levels,
        topm_per_level=args.routing_topm_per_level,
        temperature=args.routing_temp,
        phrase_weight=args.routing_phrase_weight,
        term_weight=args.routing_term_weight,
        delta_weight=args.routing_delta_weight,
    )
    debug["terms"] = terms
    debug["abstained"] = False

    raw_max_score = float(debug.get("max_gaussian_score", 0.0))
    if debug["confidence"] < args.route_conf_threshold or raw_max_score < args.route_min_raw_score:
        point_scores = torch.zeros_like(point_scores)
        debug["abstained"] = True
    return point_scores, debug


def apply_contrast(vlm, features, primary_sim, contrast_terms, contrast_weight):
    if not contrast_terms or contrast_weight <= 0:
        return primary_sim

    contrast_lists = []
    for term in contrast_terms:
        contrast_lists.append(similarity_for_query(vlm, features, term))
    adjusted = []
    for level_idx, primary in enumerate(primary_sim):
        contrast = torch.stack([contrast[level_idx] for contrast in contrast_lists], dim=0).max(dim=0).values
        adjusted.append(primary - contrast_weight * contrast)
    return adjusted


def gaussian_from_candidates(snag, candidates):
    rel_gaussians = torch.zeros(snag.gaussian_num, 1, dtype=torch.float32)
    for level, _, _, index in candidates:
        lowest_idx = torch.where(snag.labels[level] == index)[0]
        rel_gaussians[lowest_idx, 0] = 1
    return rel_gaussians


def reranked_related_gaussian(snag, primary_sim, aux_sim, levels, topk, candidate_topk, aux_weight):
    candidates = []
    for level in levels:
        sim_idx = level - 1
        primary_vals, primary_indices = torch.topk(primary_sim[sim_idx], candidate_topk)
        for primary_val, index in zip(primary_vals, primary_indices):
            aux_val = aux_sim[sim_idx][index]
            score = primary_val + aux_weight * aux_val
            candidates.append((level, score, primary_val, index))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return gaussian_from_candidates(snag, candidates[:topk])


def reranked_candidates(primary_sim, aux_sim, levels, candidate_topk, aux_weight):
    candidates = []
    for level in levels:
        sim_idx = level - 1
        primary_vals, primary_indices = torch.topk(primary_sim[sim_idx], candidate_topk)
        for primary_val, index in zip(primary_vals, primary_indices):
            aux_val = aux_sim[sim_idx][index]
            score = primary_val + aux_weight * aux_val
            candidates.append((level, score, primary_val, index))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates


def clip_crop_score(vlm, image, mask, prompt):
    ys, xs = torch.where(mask)
    if ys.numel() == 0:
        return -1e6

    pad = 8
    _, h, w = image.shape
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, h)
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, w)
    crop = image[:3, y0:y1, x0:x1].unsqueeze(0).to(vlm.device)
    crop = F.interpolate(crop, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=vlm.device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=vlm.device).view(1, 3, 1, 1)
    crop = (crop - mean) / std

    vlm.encode_text(prompt)
    with torch.no_grad():
        image_features = vlm.clip_pretrained.encode_image(crop.half()).float()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return (image_features @ vlm.text_feature[0:1].T).item()


def mask_iou(mask_a, mask_b):
    inter = torch.logical_and(mask_a, mask_b).sum().float()
    union = torch.logical_or(mask_a, mask_b).sum().float()
    return (inter / (union + 1e-6)).item()


def clip_reranked_related_gaussian(
    snag,
    candidates,
    vlm,
    gaussians,
    pipe,
    background,
    cam,
    prompt,
    topk,
    image,
    diversity_iou=0.85,
):
    ranked = []
    image = image.to(vlm.device)
    h, w = cam.image_height, cam.image_width
    for candidate in candidates:
        point_valid = gaussian_from_candidates(snag, [candidate]).expand(-1, 20).cuda()
        gaussians._semantics = point_valid
        embd_sim = render(cam, gaussians, pipe, background)["semantics"]
        mask = embd_sim.reshape(20, -1)[0].reshape(h, w) > 0.5
        ranked.append((clip_crop_score(vlm, image, mask, prompt), candidate, mask))
    ranked.sort(key=lambda x: x[0], reverse=True)

    selected = []
    selected_masks = []
    for _, candidate, mask in ranked:
        if mask.sum() == 0:
            continue
        if all(mask_iou(mask, prev_mask) < diversity_iou for prev_mask in selected_masks):
            selected.append(candidate)
            selected_masks.append(mask)
        if len(selected) >= topk:
            break

    if not selected:
        selected = [candidate for _, candidate, _ in ranked[:topk]]
    return gaussian_from_candidates(snag, selected)


def render_mask_for_candidates(snag, candidates, gaussians, pipe, background, cam):
    point_valid = gaussian_from_candidates(snag, candidates).expand(-1, 20).cuda()
    gaussians._semantics = point_valid
    embd_sim = render(cam, gaussians, pipe, background)["semantics"]
    h, w = cam.image_height, cam.image_width
    return embd_sim.reshape(20, -1)[0].reshape(h, w) > 0.5


def candidate_score_value(candidate):
    score = candidate[1]
    if torch.is_tensor(score):
        return float(score.detach().cpu())
    return float(score)


def plan_anchor_mask(snag, vlm, gaussians, pipe, background, cam, anchor_terms, args):
    if not anchor_terms:
        return None
    anchor_sim = similarity_for_terms(vlm, snag.feat, anchor_terms)
    anchor_candidates = reranked_candidates(
        anchor_sim,
        anchor_sim,
        args.anchor_levels,
        args.anchor_candidate_topk,
        0.0,
    )[:args.anchor_topk]
    if not anchor_candidates:
        return None
    return render_mask_for_candidates(snag, anchor_candidates, gaussians, pipe, background, cam)


def anchor_filtered_related_gaussian(
    snag,
    candidates,
    gaussians,
    pipe,
    background,
    cam,
    topk,
    anchor_mask,
    containment_threshold,
    containment_weight,
    area_penalty,
    diversity_iou,
):
    h, w = cam.image_height, cam.image_width
    image_area = max(h * w, 1)
    ranked = []

    for candidate in candidates:
        mask = render_mask_for_candidates(snag, [candidate], gaussians, pipe, background, cam)
        area = float(mask.sum().item())
        if area == 0:
            continue

        containment = 1.0
        if anchor_mask is not None:
            containment = float(torch.logical_and(mask, anchor_mask).sum().item() / (area + 1e-6))
            if containment < containment_threshold:
                continue

        area_ratio = area / image_area
        score = candidate_score_value(candidate) + containment_weight * containment - area_penalty * area_ratio
        ranked.append((score, candidate, mask))

    ranked.sort(key=lambda x: x[0], reverse=True)
    selected = []
    selected_masks = []
    for _, candidate, mask in ranked:
        if all(mask_iou(mask, prev_mask) < diversity_iou for prev_mask in selected_masks):
            selected.append(candidate)
            selected_masks.append(mask)
        if len(selected) >= topk:
            break

    if not selected:
        selected = [candidate for _, candidate, _ in ranked[:topk]] or candidates[:topk]
    return gaussian_from_candidates(snag, selected)


def proposal_reasoned_gaussian(
    proposal_reasoner,
    image_tensor,
    candidates,
    snag,
    gaussians,
    pipe,
    background,
    cam,
    plan,
    scene,
    frame,
    prompt,
    candidate_topk,
):
    if proposal_reasoner is None:
        return None
    image_rgb = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    masks = []
    metadata = []
    for idx, candidate in enumerate(candidates[:candidate_topk], start=1):
        mask = render_mask_for_candidates(snag, [candidate], gaussians, pipe, background, cam)
        mask_np = mask.detach().cpu().numpy().astype(bool)
        area = int(mask_np.sum())
        if area == 0:
            continue
        masks.append(mask_np)
        metadata.append({
            "id": idx,
            "level": int(candidate[0]),
            "superpoint_id": int(candidate[3].detach().cpu()) if torch.is_tensor(candidate[3]) else int(candidate[3]),
            "score": candidate_score_value(candidate),
            "area": area,
        })
    selected_ids = proposal_reasoner.select(
        image_rgb,
        masks,
        metadata,
        plan,
        scene=scene,
        frame=frame,
        query=prompt,
    )
    if not selected_ids:
        return None
    selected_candidates = []
    for selected_id in selected_ids:
        for meta, candidate in zip(metadata, candidates[:candidate_topk]):
            if meta["id"] == selected_id:
                selected_candidates.append(candidate)
                break
    if not selected_candidates:
        return None
    return gaussian_from_candidates(snag, selected_candidates[: args.proposal_select_topk])

def polygon_to_mask(img_shape, points_list):
    points = np.asarray(points_list, dtype=np.int32)
    mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    return mask

@torch.no_grad()
def training(dataset, pipe):
    gaussians = GaussianModel(dataset.sh_degree, 20)
    scene = Scene(dataset, gaussians, 30000, load_sem=False)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    nag = torch.load(os.path.join(dataset.model_path, f"sai_nag.pt"))

    vlm = ClipSimMeasure()
    vlm.load_model()
    snag = SemanticNAG(nag['nag'], nag['nag_feat'])
    route_features = snag.feat
    if args.semantic_mode == "hier_soft":
        route_features = snag.get_hierarchy_features(
            parent_weight=args.hier_parent_weight,
            residual_weight=args.hier_residual_weight,
            gate_center=args.hier_gate_center,
            gate_temp=args.hier_gate_temp,
        )
    reasoner = QueryReasoner(
        backend=args.reasoner_backend,
        cache_dir=args.reasoner_cache_dir,
        model=args.reasoner_model,
        enabled=args.query_reasoning,
    )
    proposal_reasoner = None
    if args.proposal_reasoning:
        proposal_reasoner = ProposalReasoner(
            backend=args.proposal_backend,
            model=args.proposal_model,
            cache_dir=args.proposal_cache_dir,
        )

    # "[scene]/[prompt]/[colmap_format_dataset]"
    scene_name = dataset.source_path.split('/')[-1]
    data_path = os.path.join(os.path.dirname(dataset.source_path), 'label', scene_name)
    out_path = os.path.join(args.path_pred, scene_name)
    os.makedirs(out_path, exist_ok=True)
    # img_list = os.listdir(data_path) find ends with .jpg
    img_list = [f for f in os.listdir(data_path) if f.endswith('.jpg')]
    for im in img_list:
        image_name = im.split('.')[0]
        js_file = os.path.join(data_path, image_name+'.json')
        anno = json.load(open(js_file))
        image_np = cv2.imread(os.path.join(data_path, im), cv2.IMREAD_COLOR)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image_np).float().permute(2, 0, 1) / 255.0
        for cam in scene.getTrainCameras():
            if cam.image_name == image_name:
                break

        os.makedirs(os.path.join(out_path, cam.image_name), exist_ok=True)
        prompt_list = [obj['category'] for obj in anno['objects']]
        prompt_list = list(set(prompt_list))
        frame_debug = {}

        for prompt in prompt_list:
            image_path = os.path.join(data_path, im)
            plan = reasoner.plan(
                prompt,
                candidate_labels=prompt_list,
                image_path=image_path,
                scene=scene_name,
                frame=image_name,
            )
            # segmentation prediction
            use_soft_route = should_soft_route(prompt, plan, args)
            if use_soft_route:
                point_valid, route_debug = soft_routed_gaussian(snag, vlm, route_features, prompt, args)
                frame_debug[prompt] = route_debug
            elif args.refine_parts and plan.role == "modifier":
                target_terms = plan.target_terms or [prompt]
                target_weights = [1.0] + [args.expanded_prompt_weight] * (len(target_terms) - 1)
                primary_sim = similarity_for_terms(vlm, snag.feat, target_terms, target_weights)
                primary_sim = apply_contrast(
                    vlm, snag.feat, primary_sim, plan.contrast_terms, args.contrast_weight
                )
                aux_terms = plan.part_terms or prompt_parts(prompt)
                aux_weights = [1.0] + [args.expanded_prompt_weight] * (len(aux_terms) - 1)
                aux_sim = similarity_for_terms(vlm, snag.feat, aux_terms, aux_weights)
                if args.modifier_clip_rerank:
                    candidates = reranked_candidates(
                        primary_sim,
                        aux_sim,
                        args.modifier_levels,
                        args.modifier_candidate_topk,
                        args.modifier_aux_weight,
                    )
                    point_valid = proposal_reasoned_gaussian(
                        proposal_reasoner,
                        image_tensor,
                        candidates,
                        snag,
                        gaussians,
                        pipe,
                        background,
                        cam,
                        plan,
                        scene_name,
                        image_name,
                        prompt,
                        args.proposal_candidate_topk,
                    )
                    if point_valid is None:
                        point_valid = clip_reranked_related_gaussian(
                            snag,
                            candidates,
                            vlm,
                            gaussians,
                            pipe,
                            background,
                            cam,
                            prompt,
                            args.modifier_topk,
                            image_tensor,
                            args.modifier_diversity_iou,
                        )
                else:
                    point_valid = reranked_related_gaussian(
                        snag,
                        primary_sim,
                        aux_sim,
                        args.modifier_levels,
                        args.modifier_topk,
                        args.modifier_candidate_topk,
                        args.modifier_aux_weight,
                    )
            elif args.refine_parts and plan.role == "part":
                target_terms = plan.target_terms or [prompt]
                target_weights = [1.0] + [args.expanded_prompt_weight] * (len(target_terms) - 1)
                primary_sim = similarity_for_terms(vlm, snag.feat, target_terms, target_weights)
                primary_sim = apply_contrast(
                    vlm, snag.feat, primary_sim, plan.contrast_terms, args.contrast_weight
                )
                aux_terms = plan.part_terms or prompt_parts(prompt)
                aux_weights = [1.0] + [args.expanded_prompt_weight] * (len(aux_terms) - 1)
                aux_sim = similarity_for_terms(vlm, snag.feat, aux_terms, aux_weights)
                candidates = reranked_candidates(
                    primary_sim,
                    aux_sim,
                    args.part_levels,
                    args.part_candidate_topk,
                    args.part_aux_weight,
                )
                point_valid = proposal_reasoned_gaussian(
                    proposal_reasoner,
                    image_tensor,
                    candidates,
                    snag,
                    gaussians,
                    pipe,
                    background,
                    cam,
                    plan,
                    scene_name,
                    image_name,
                    prompt,
                    args.proposal_candidate_topk,
                )
                if point_valid is not None:
                    point_valid = point_valid
                else:
                    anchor_mask = plan_anchor_mask(
                        snag, vlm, gaussians, pipe, background, cam, plan.anchor_terms, args
                    )
                    point_valid = anchor_filtered_related_gaussian(
                        snag,
                        candidates,
                        gaussians,
                        pipe,
                        background,
                        cam,
                        args.part_topk,
                        anchor_mask,
                        args.part_anchor_min_containment,
                        args.part_anchor_weight,
                        args.part_area_penalty,
                        args.part_diversity_iou,
                    )
            else:
                vlm.encode_text(prompt)
                point_valid = snag.get_related_gaussian(
                    [vlm.compute_similarity(f) for f in snag.feat],
                    topk=args.topk,
                    level=args.levels,
                )
            point_valid = point_valid.expand(-1, 20).cuda()
            gaussians._semantics = point_valid        
            embd_sim = render(cam, gaussians, pipe, background)["semantics"]
            w, h = cam.image_width, cam.image_height
            semantic_score = embd_sim.reshape(20, -1)[0]
            if use_soft_route:
                rendered_max = semantic_score.max()
                if prompt in frame_debug:
                    frame_debug[prompt]["rendered_max"] = float(rendered_max.detach().cpu())
                if rendered_max > 1e-6:
                    semantic_score = semantic_score / rendered_max
                mask = semantic_score > args.route_mask_threshold
            else:
                mask = semantic_score > 0.5
            binary_mask = mask.reshape(h, w)

            # get ground truth mask
            mask_gt = np.zeros((h, w), dtype=np.uint8)
            for obj in anno['objects']:
                if obj['category'] == prompt:
                    _mask_gt = polygon_to_mask((h, w), obj['segmentation'])
                    mask_gt = np.maximum(mask_gt, _mask_gt)

            cv2.imwrite(os.path.join(out_path, cam.image_name, prompt.replace(' ', '_')+'.png'), binary_mask.cpu().numpy() * 255)
            cv2.imwrite(os.path.join(out_path, cam.image_name, prompt.replace(' ', '_')+'_gt.png'), mask_gt * 255)
            if prompt in frame_debug:
                frame_debug[prompt]["rendered_area"] = int(binary_mask.sum().detach().cpu())

        if args.semantic_mode in ["soft", "hier_soft"]:
            debug_dir = os.path.join(args.route_debug_dir, scene_name)
            os.makedirs(debug_dir, exist_ok=True)
            with open(os.path.join(debug_dir, cam.image_name + ".json"), "w", encoding="utf-8") as f:
                json.dump(frame_debug, f, indent=2)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--path_pred', type=str, default=None)
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--levels', nargs='+', type=int, default=[2, 3])
    parser.add_argument('--semantic_mode', type=str, default='hard', choices=['hard', 'soft', 'hier_soft'])
    parser.add_argument('--routing_levels', nargs='+', type=int, default=[2, 3])
    parser.add_argument('--routing_temp', type=float, default=0.07)
    parser.add_argument('--routing_topm_per_level', type=int, default=24)
    parser.add_argument('--routing_phrase_weight', type=float, default=1.0)
    parser.add_argument('--routing_term_weight', type=float, default=0.35)
    parser.add_argument('--routing_delta_weight', type=float, default=0.5)
    parser.add_argument('--hier_parent_weight', type=float, default=0.25)
    parser.add_argument('--hier_residual_weight', type=float, default=0.15)
    parser.add_argument('--hier_gate_center', type=float, default=0.2)
    parser.add_argument('--hier_gate_temp', type=float, default=0.07)
    parser.add_argument('--route_conf_threshold', type=float, default=0.12)
    parser.add_argument('--route_min_raw_score', type=float, default=0.0)
    parser.add_argument('--route_mask_threshold', type=float, default=0.3)
    parser.add_argument('--route_debug_dir', type=str, default='output/debug_hier_soft')
    parser.add_argument('--soft_route_roles', nargs='+', default=['part', 'modifier'], choices=['object', 'part', 'modifier'])
    parser.add_argument('--refine_parts', action='store_true')
    parser.add_argument('--part_topk', type=int, default=2)
    parser.add_argument('--part_levels', nargs='+', type=int, default=[1, 2])
    parser.add_argument('--part_candidate_topk', type=int, default=8)
    parser.add_argument('--part_aux_weight', type=float, default=2.0)
    parser.add_argument('--part_anchor_min_containment', type=float, default=0.2)
    parser.add_argument('--part_anchor_weight', type=float, default=0.5)
    parser.add_argument('--part_area_penalty', type=float, default=0.25)
    parser.add_argument('--part_diversity_iou', type=float, default=0.85)
    parser.add_argument('--anchor_topk', type=int, default=3)
    parser.add_argument('--anchor_levels', nargs='+', type=int, default=[2, 3])
    parser.add_argument('--anchor_candidate_topk', type=int, default=8)
    parser.add_argument('--modifier_topk', type=int, default=3)
    parser.add_argument('--modifier_levels', nargs='+', type=int, default=[2, 3])
    parser.add_argument('--modifier_candidate_topk', type=int, default=8)
    parser.add_argument('--modifier_aux_weight', type=float, default=0.25)
    parser.add_argument('--modifier_clip_rerank', action='store_true')
    parser.add_argument('--modifier_diversity_iou', type=float, default=0.85)
    parser.add_argument('--query_reasoning', action='store_true')
    parser.add_argument('--reasoner_backend', type=str, default='heuristic', choices=['heuristic', 'openai'])
    parser.add_argument('--reasoner_model', type=str, default='gpt-4o')
    parser.add_argument('--reasoner_cache_dir', type=str, default='output/query_reasoning')
    parser.add_argument('--contrast_weight', type=float, default=0.1)
    parser.add_argument('--expanded_prompt_weight', type=float, default=0.35)
    parser.add_argument('--proposal_reasoning', action='store_true')
    parser.add_argument('--proposal_backend', type=str, default='heuristic', choices=['heuristic', 'openai'])
    parser.add_argument('--proposal_model', type=str, default='gpt-4o')
    parser.add_argument('--proposal_cache_dir', type=str, default='output/proposal_reasoning')
    parser.add_argument('--proposal_candidate_topk', type=int, default=8)
    parser.add_argument('--proposal_select_topk', type=int, default=2)
    args = parser.parse_args(sys.argv[1:])
    if args.path_pred is None:
        if args.semantic_mode == 'soft':
            args.path_pred = 'output/render_soft/lerf'
        elif args.semantic_mode == 'hier_soft':
            args.path_pred = 'output/render_hier_soft/lerf'
        else:
            args.path_pred = 'output/render/lerf'
    if args.semantic_mode == 'soft' and args.route_debug_dir == 'output/debug_hier_soft':
        args.route_debug_dir = 'output/debug_soft'

    safe_state(True)
    training(lp.extract(args), pp.extract(args))

    # All done
    print("\nPred complete.")
