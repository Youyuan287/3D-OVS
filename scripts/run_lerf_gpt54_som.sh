#!/usr/bin/env bash
set -euo pipefail

# Run from an already patched THGS-main checkout.
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-/home/Groups/group2/Working/tyy/data/lerf_ovs}"
LABEL_ROOT="${LABEL_ROOT:-$DATA_ROOT/label}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-output/gpt54_som_low_all_artifacts}"
BASELINE_PRED="${BASELINE_PRED:-output/render/lerf}"
GPT_PRED="${GPT_PRED:-output/render_gpt54_som_low_all/lerf}"
ORACLE_PRED="${ORACLE_PRED:-output/render_gpt54_som_low_all_oracle/lerf}"
SCENES="${SCENES:-figurines ramen teatime waldo_kitchen}"

for scene in $SCENES; do
  python llm_som_object_reroute.py build \
    --project_root "$PROJECT_ROOT" \
    --data_root "$DATA_ROOT" \
    --label_root "$LABEL_ROOT" \
    --artifact_dir "$ARTIFACT_ROOT/$scene" \
    --scene "$scene" \
    --all_prompts \
    --low_iou_threshold 0.10
done

cat <<'MSG'
Build complete.

Next step for GPT-5.4 reproduction:
1. Create selections_gpt54.jsonl under each artifact scene directory.
2. The expected JSONL schema is one row per task with:
   scene, frame, target, selected_ids, selected_iou, raw_response
3. Keep API keys in environment variables only. Do not commit them.

Then apply/report:
MSG

for scene in $SCENES; do
  python llm_som_object_reroute.py apply \
    --project_root "$PROJECT_ROOT" \
    --artifact_dir "$ARTIFACT_ROOT/$scene" \
    --scene "$scene" \
    --baseline_pred "$BASELINE_PRED" \
    --out_pred "$GPT_PRED" \
    --selection_file selections_gpt54.jsonl \
    --mode gpt54

  python llm_som_object_reroute.py apply \
    --project_root "$PROJECT_ROOT" \
    --artifact_dir "$ARTIFACT_ROOT/$scene" \
    --scene "$scene" \
    --baseline_pred "$BASELINE_PRED" \
    --out_pred "$ORACLE_PRED" \
    --mode oracle
done

python scripts/eval_seg.py \
  --dataset lerf \
  --path_gt "$LABEL_ROOT" \
  --path_pred "$GPT_PRED" \
  --scene_list figurines ramen teatime waldo_kitchen

python scripts/eval_seg.py \
  --dataset lerf \
  --path_gt "$LABEL_ROOT" \
  --path_pred "$ORACLE_PRED" \
  --scene_list figurines ramen teatime waldo_kitchen
