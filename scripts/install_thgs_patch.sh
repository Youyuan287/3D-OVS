#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/install_thgs_patch.sh /path/to/THGS-main" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
thgs_root="$1"

if [[ ! -d "$thgs_root" ]]; then
  echo "THGS checkout not found: $thgs_root" >&2
  exit 1
fi

mkdir -p "$thgs_root/scripts"

cp "$repo_root/remote_thgs_patch/test_lerf.py" "$thgs_root/test_lerf.py"
cp "$repo_root/remote_thgs_patch/query_reasoner.py" "$thgs_root/query_reasoner.py"
cp "$repo_root/remote_thgs_patch/proposal_reasoner.py" "$thgs_root/proposal_reasoner.py"
cp "$repo_root/remote_thgs_patch/nag_data.py" "$thgs_root/nag_data.py"
cp "$repo_root/remote_thgs_patch/parent_part_proposal_render.py" "$thgs_root/parent_part_proposal_render.py"
cp "$repo_root/remote_thgs_patch/diagnose_parent_part_candidates.py" "$thgs_root/diagnose_parent_part_candidates.py"
cp "$repo_root/remote_thgs_patch/merge_scene_fill_soft_roles.py" "$thgs_root/merge_scene_fill_soft_roles.py"
cp "$repo_root/remote_thgs_patch/llm_som_object_reroute.py" "$thgs_root/llm_som_object_reroute.py"
cp "$repo_root/remote_thgs_patch/scripts/eval_seg.py" "$thgs_root/scripts/eval_seg.py"

echo "Installed THGS patches into: $thgs_root"
