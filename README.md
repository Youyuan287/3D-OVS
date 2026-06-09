# 3D-OVS

THGS improvement patches and evaluation utilities for fine-grained open-vocabulary 3D Gaussian segmentation experiments.

This repository stores the lightweight code needed to reproduce the THGS parent-part routing and GPT/Qwen SoM rerouting experiments. It is intended to be applied on top of a working THGS checkout with LERF-OVS data, trained Gaussian checkpoints, language feature maps, and local/API VLM credentials prepared separately.

## Contents

- `REPRODUCE_THGS.md`: end-to-end reproduction notes, required external assets, install commands, and evaluation commands.
- `scripts/install_thgs_patch.sh`: copy this repository's patch files into an existing THGS checkout.
- `scripts/run_lerf_gpt54_som.sh`: command template for building/applying/reporting four-scene GPT-SoM rerouting results.
- `results/lerf_gpt54_som_summary.json`: recorded headline metrics for the stored experiment version.
- `remote_thgs_patch/test_lerf.py`: THGS LERF inference with hierarchy-aware soft routing and role-gated routing.
- `remote_thgs_patch/query_reasoner.py`: prompt role planner for object / part / modifier routing.
- `remote_thgs_patch/proposal_reasoner.py`: optional SoM-style proposal selector and debug panel writer for part proposals.
- `remote_thgs_patch/nag_data.py`: SemanticNAG helper with hierarchy feature construction used by `test_lerf.py`.
- `remote_thgs_patch/parent_part_proposal_render.py`: parent-conditioned fine proposal selector for part queries.
- `remote_thgs_patch/diagnose_parent_part_candidates.py`: candidate-level diagnostic tool for parent-part selection.
- `remote_thgs_patch/merge_scene_fill_soft_roles.py`: helper for merging role-specific outputs.
- `remote_thgs_patch/scripts/eval_seg.py`: LERF evaluation script using polygon annotations.

## Quick Start

```bash
git clone https://github.com/Youyuan287/3D-OVS.git
cd 3D-OVS

# Apply patches to an existing THGS checkout.
bash scripts/install_thgs_patch.sh /path/to/THGS-main

# Then run commands from inside /path/to/THGS-main.
```

See `REPRODUCE_THGS.md` for the full checklist.
