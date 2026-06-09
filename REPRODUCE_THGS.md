# THGS Reproduction Notes

This document describes how to reproduce the THGS parent-part routing and SoM rerouting experiments from this repository on another machine.

The repository does not vendor the original THGS project, LERF-OVS data, trained Gaussian checkpoints, model weights, or generated output images. Those assets are large and should be prepared outside Git.

## 1. Required External Assets

Prepare these paths before running the scripts.

| Asset | Expected content | Example path on 4090 server |
|---|---|---|
| THGS checkout | Original THGS codebase with Gaussian renderer, scene loader, arguments, utils, ext/spt, etc. | `/home/Groups/group2/Working/tyy/project/THGS-main` |
| LERF-OVS data | RGB frames and polygon labels under `label/<scene>/frame_*.jpg/json` | `/home/Groups/group2/Working/tyy/data/lerf_ovs` |
| Fine segment features | `language_features_part_only_update/<frame>_s.npy` and `<frame>_meta.json` | inside each LERF scene directory |
| Trained THGS checkpoints | Gaussian model output and `sai_nag.pt` used by `test_lerf.py` | THGS model/output directories |
| Baseline predictions | baseline masks under `output/render/lerf/<scene>/<frame>/<prompt>.png` | `output/render/lerf` |
| Optional proposal predictions | `render_proposal_clip_cosine_k1/k2/k3`, `render_scene_opt` | `output/...` |
| Qwen3-VL model | local model for open-source VLM selection | `/home/Groups/group2/.cache/modelscope/hub/models/Qwen/Qwen3-VL-8B-Instruct` |
| GPT API key | optional closed VLM selection, kept only in env vars | `OPENAI_API_KEY` |

Large generated artifacts are intentionally excluded from Git:

```text
output/
data/
models/
*.pth
*.pt
*.ckpt
```

## 2. Apply This Patch Repo to THGS

```bash
git clone https://github.com/Youyuan287/3D-OVS.git
cd 3D-OVS
bash scripts/install_thgs_patch.sh /path/to/THGS-main
```

This copies the following files into the THGS checkout:

```text
test_lerf.py
query_reasoner.py
proposal_reasoner.py
nag_data.py
parent_part_proposal_render.py
diagnose_parent_part_candidates.py
merge_scene_fill_soft_roles.py
llm_som_object_reroute.py
scripts/eval_seg.py
```

## 3. Environment

Use the original THGS environment, plus the packages required by the new utilities:

```bash
pip install numpy opencv-python pillow tqdm pandas
pip install openai
```

For Qwen3-VL inference, use an environment with a recent PyTorch/transformers stack that can load `Qwen3-VL-8B-Instruct`.

Keep API keys out of Git:

```bash
export OPENAI_API_KEY="..."
```

## 4. Baseline THGS Inference

Run from the patched THGS checkout.

```bash
python test_lerf.py \
  -m /path/to/trained/thgs/model \
  -s /path/to/lerf_ovs/<scene> \
  --path_pred output/render/lerf
```

The patched `test_lerf.py` also supports hierarchy-aware and parent-part options:

```bash
python test_lerf.py \
  -m /path/to/trained/thgs/model \
  -s /path/to/lerf_ovs/<scene> \
  --semantic_mode hier_soft \
  --refine_parts \
  --query_reasoning \
  --path_pred output/render_hier_soft/lerf
```

The parent-part link is implemented through:

```text
query_reasoner.py -> role / anchor_terms / part_terms
nag_data.py -> child_to_parent and hierarchy-aware features
test_lerf.py -> anchor_mask containment filtering
```

## 5. Parent-Part Fine Segment Post-Processing

After baseline predictions and `language_features_part_only_update` are available:

```bash
python parent_part_proposal_render.py \
  --data_root /path/to/lerf_ovs \
  --label_root /path/to/lerf_ovs/label \
  --baseline_pred output/render/lerf \
  --out_root output/render_parent_part/lerf \
  --scenes figurines ramen teatime waldo_kitchen
```

Diagnostics:

```bash
python diagnose_parent_part_candidates.py
```

Outputs:

```text
output/render_parent_part/lerf
output/parent_part_debug.json
```

## 6. SoM Rerouting Experiment

Build SoM candidate panels for the four LERF scenes:

```bash
for scene in figurines ramen teatime waldo_kitchen; do
  python llm_som_object_reroute.py build \
    --project_root . \
    --data_root /path/to/lerf_ovs \
    --label_root /path/to/lerf_ovs/label \
    --artifact_dir output/gpt54_som_low_all_artifacts/$scene \
    --scene $scene \
    --all_prompts \
    --low_iou_threshold 0.10
done
```

Qwen3-VL selection:

```bash
python llm_som_object_reroute.py qwen \
  --artifact_dir output/gpt54_som_low_all_artifacts/ramen \
  --scene ramen \
  --model /path/to/Qwen3-VL-8B-Instruct \
  --selection_file selections_qwen.jsonl
```

GPT selection was run with an OpenAI-compatible API. The raw API key is not stored in this repository. Save the result as:

```text
output/gpt54_som_low_all_artifacts/<scene>/selections_gpt54.jsonl
```

Apply GPT selections:

```bash
for scene in figurines ramen teatime waldo_kitchen; do
  python llm_som_object_reroute.py apply \
    --project_root . \
    --artifact_dir output/gpt54_som_low_all_artifacts/$scene \
    --scene $scene \
    --baseline_pred output/render/lerf \
    --out_pred output/render_gpt54_som_low_all/lerf \
    --selection_file selections_gpt54.jsonl \
    --mode gpt54
done
```

Apply oracle candidate upper bound:

```bash
for scene in figurines ramen teatime waldo_kitchen; do
  python llm_som_object_reroute.py apply \
    --project_root . \
    --artifact_dir output/gpt54_som_low_all_artifacts/$scene \
    --scene $scene \
    --baseline_pred output/render/lerf \
    --out_pred output/render_gpt54_som_low_all_oracle/lerf \
    --mode oracle
done
```

## 7. Evaluation

```bash
python scripts/eval_seg.py \
  --dataset lerf \
  --path_gt /path/to/lerf_ovs/label \
  --path_pred output/render/lerf \
  --scene_list figurines ramen teatime waldo_kitchen

python scripts/eval_seg.py \
  --dataset lerf \
  --path_gt /path/to/lerf_ovs/label \
  --path_pred output/render_gpt54_som_low_all/lerf \
  --scene_list figurines ramen teatime waldo_kitchen
```

Recorded metrics for this code snapshot are stored in:

```text
results/lerf_gpt54_som_summary.json
```

Headline result:

```text
baseline THGS mIoU: 0.5885
GPT-5.4 SoM rerouting mIoU: 0.6748
oracle candidate upper bound mIoU: 0.7362
```

## 8. Visual Artifacts

Expected artifact locations:

```text
output/gpt54_som_low_all_artifacts/<scene>/som_panels/
output/gpt54_som_low_all_artifacts/<scene>/applied_compare/gpt54/
output/gpt54_som_low_all_artifacts/<scene>/applied_compare/oracle/
output/render_gpt54_som_low_all/lerf
output/render_gpt54_som_low_all_oracle/lerf
```

The original 4090 server paths used for the report were:

```text
/home/Groups/group2/Working/tyy/project/THGS-main/output/gpt54_som_low_all_artifacts/
/home/Groups/group2/Working/tyy/project/THGS-main/output/render_gpt54_som_low_all/lerf
/home/Groups/group2/Working/tyy/project/THGS-main/output/render_gpt54_som_low_all_oracle/lerf
```

## 9. Current Limitation

This repository is a reproducibility patch package, not a full mirror of THGS and its datasets. To reproduce from a fresh machine, first install the original THGS project and prepare the external assets listed in Section 1, then apply this repository's patch files.
