# 3D-OVS

THGS improvement patches and evaluation utilities for fine-grained open-vocabulary 3D Gaussian segmentation experiments.

## Contents

- `remote_thgs_patch/test_lerf.py`: THGS LERF inference with hierarchy-aware soft routing and role-gated routing.
- `remote_thgs_patch/query_reasoner.py`: prompt role planner for object / part / modifier routing.
- `remote_thgs_patch/proposal_reasoner.py`: optional SoM-style proposal selector and debug panel writer for part proposals.
- `remote_thgs_patch/nag_data.py`: SemanticNAG helper with hierarchy feature construction used by `test_lerf.py`.
- `remote_thgs_patch/parent_part_proposal_render.py`: parent-conditioned fine proposal selector for part queries.
- `remote_thgs_patch/diagnose_parent_part_candidates.py`: candidate-level diagnostic tool for parent-part selection.
- `remote_thgs_patch/merge_scene_fill_soft_roles.py`: helper for merging role-specific outputs.
- `remote_thgs_patch/scripts/eval_seg.py`: LERF evaluation script using polygon annotations.
