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
    teatime_bear_nose_*.png
    teatime_hooves_*.png
```

The report `THGS_GPT_SoM_Rerouting_汇报总结.md` references these files. If the images are missing after clone, copy them from:

```text
/home/Groups/group2/Working/tyy/project/THGS-main/output/gpt54_som_low_all_artifacts/
/home/Groups/group2/Working/tyy/project/THGS-main/output/render_parent_part/lerf
```
