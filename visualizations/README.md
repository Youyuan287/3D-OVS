# Visualization Assets

This directory is reserved for representative THGS visualization results copied from the 4090 server.

Expected structure:

```text
visualizations/
  som_panels/
    ramen_bowl_frame_00006_som.png
    ramen_plate_frame_*.png
    ramen_sake_cup_frame_*.png
    ramen_kamaboko_frame_*.png
    ramen_corn_frame_*.png
    ramen_onion_segments_frame_*.png
  gpt54_compare/
    ramen_bowl_frame_00006.png
    ramen_plate_frame_*.png
    ramen_sake_cup_frame_*.png
    ramen_kamaboko_frame_*.png
    ramen_corn_frame_*.png
    ramen_onion_segments_frame_*.png
  oracle_compare/
    representative_oracle_upper_bound.png
  parent_part/
    teatime_frame_00002_bear_nose_parent_part_compare.png
    teatime_frame_00025_hooves_parent_part_compare.png
  som_part_cases/
    teatime_bear_nose_frame_00002_*.png
    teatime_hooves_frame_00025_*.png
```

The report `THGS_GPT_SoM_Rerouting_汇报总结.md` references these files. If the images are missing after clone, copy them from:

```text
/home/Groups/group2/Working/tyy/project/THGS-main/output/gpt54_som_low_all_artifacts/
/home/Groups/group2/Working/tyy/project/THGS-main/output/render_parent_part/lerf
```

Important distinction:

- `parent_part/` contains true `render_parent_part` outputs for evaluating the parent-part module.
- `som_part_cases/` contains GPT-SoM rerouting diagnostics for part queries. These should not be used as evidence that the parent-part module itself works.
