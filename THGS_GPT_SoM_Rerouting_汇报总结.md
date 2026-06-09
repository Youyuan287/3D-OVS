# THGS 层级语义改进与 GPT-SoM Rerouting 汇报总结

日期：2026-06-08  
项目：开放词汇 3D Gaussian Splatting 语义分割 / THGS  
远程实验服务器：4090 服务器  
远程 THGS 目录：

```text
/home/Groups/group2/Working/tyy/project/THGS-main
```

## 1. 交流与工作背景

本轮讨论围绕 THGS 在 LERF-OVS 四个场景上的开放词汇语义分割效果展开。前期已经尝试过层级语义继承、软查询路由、parent-part 语义约束、scene fill 等方案，但整体指标提升有限。

用户明确指出：

- `scene fill` 过于工程化，不适合作为主要方法。
- 需要分析 parent-part 语义目标是否真正达到。
- 当前指标只比 baseline 高一点，需要进一步分析大目标 object 类提升不明显的原因。
- 可以参考 ReLaGS 借助大模型提升 object segmentation 效果。
- 重点先在 `ramen` 的 object 低分样例上验证 LLM-SoM rerouting 是否有效。

因此，本轮工作的核心问题转为：

> THGS 的 object-level failure 是因为候选 mask 不存在，还是因为已有候选没有被正确路由/选择？

## 2. 初始问题分析

对 baseline 和已有改进结果分析后，发现：

- 当前非 fill 方法主要改善 part 类别。
- object 类别整体提升不明显。
- 在 `ramen` 场景中，低 IoU object 样例高度集中。

典型失败包括：

| 类型 | 示例 |
|---|---|
| 空 mask | `bowl`, `plate`, `sake cup` |
| 大面积误检 | `kamaboko`, `corn` |
| 细小目标选错 | `onion segments` |
| 候选存在但语义标签错误 | `bowl` 的正确 segment 被标成 `plastic ladle` |

关键发现：

> 许多正确 mask 已经存在于 fine segment map 中，但 CLIP 文本标签或原始语义路由没有把它选出来。

因此，问题更像是 **candidate rerouting / candidate selection 不足**，而不是单纯的候选缺失。

## 3. Parent-part 语义继承与细粒度 part 修正

在转向 SoM rerouting 之前，先审查了本地已有的“THGS 层级语义继承与软查询路由改进方案”。这条线的目标不是直接提升所有 object mask，而是解决细粒度 part prompt 在 THGS 中容易被父物体淹没的问题。

### 3.1 原始 parent-part 目标

parent-part 设计目标可以概括为：

> 对 `bear nose`, `hooves`, `hand`, `hat` 等 part 查询，不再只按文本相似度在全场景中选 Gaussian，而是先找到父物体或锚点区域，再在该区域内部选择更小、更语义匹配的 part segment。

具体希望达到三点：

1. **继承父物体语义上下文**：例如 `bear nose` 应该优先在 `stuffed bear` 区域附近寻找，而不是在整张图中寻找任意 “nose-like” 或小区域。
2. **抑制 part 扩散成 object**：例如 `hooves` 不能直接选中整只羊或整只玩偶，而应该只覆盖父物体中的局部区域。
3. **提升细粒度 part 稳定性**：同一个 part 在相邻层级 segment 中可能有多个候选，需要利用 containment、面积先验和跨层一致性做选择。

### 3.2 效果不佳的原因

审查发现，早期 parent-part 改动整体提升有限，主要有以下原因：

| 问题 | 影响 |
|---|---|
| 只靠文本相似度选 part | CLIP 容易把父物体整体或相邻部件打高分，part mask 过大 |
| parent 约束不够硬 | `bear nose` 等查询仍可能漂移到父物体外部的相似区域 |
| 面积先验缺失 | 小 part 和大 object 在得分上没有足够区分 |
| 候选层级不稳定 | 不同 NAG/segment level 中同一 part 的边界质量不同 |
| 对 object 类帮助有限 | parent-part 针对 part，不解决 `bowl`, `plate`, `sake cup` 这类 object 选错候选的问题 |

因此，parent-part 的目的只“部分达到”：它能让若干 part 查询更符合父物体局部约束，但不能解释或解决四场景中 object-level mIoU 提升不高的问题。

### 3.3 Parent 和 part 在代码中的链接方式

本项目中 parent-part 不是只在文字上描述“父物体-部件关系”，而是在代码中显式落到三个对象：

| 代码对象 | 含义 | 例子 |
|---|---|---|
| `role` | 当前 prompt 的类型 | `object`, `part`, `modifier` |
| `anchor_terms` | parent/base object，用来生成空间锚点 | `bear nose` 的 anchor 是 `bear` 或候选中的 bear 相关 label |
| `part_terms` | part noun 或局部语义，用来重排细粒度候选 | `bear nose` 的 part terms 是 `nose`, `bear nose` |

入口在：

```text
remote_thgs_patch/query_reasoner.py
```

`heuristic_query_plan()` 会先解析 prompt：

```text
query = "bear nose"
tail = "nose"
tail in PART_HEADS -> role = "part"
parent = "bear"
target_terms = ["bear nose", "nose", "nose of bear", "bear's nose"]
anchor_terms = ["bear"]
part_terms = ["nose", "bear nose"]
```

如果 prompt 是 `object with modifier` 结构，例如 `figurine with red hat`，则会被解析为：

```text
role = "modifier"
anchor_terms = [base object]
part_terms = [modifier phrase, modifier head]
contrast_terms = [same base object but different modifier]
```

所以 parent-part 链接的第一步是：**把自然语言 prompt 拆成 parent/base anchor 和 part/local term**。

### 3.4 层级语义继承如何实现

层级语义继承在：

```text
remote_thgs_patch/nag_data.py
```

`SemanticNAG` 会根据多层 superpoint label 建立 child-parent 映射：

```text
build_parent_child_maps(labels)
child_to_parent[level][child_superpoint_id] = parent_superpoint_id
```

也就是说，每个低层细粒度 superpoint 都知道自己属于哪个高层 parent superpoint。随后 `get_hierarchy_features()` 对低层 feature 加入 parent 上下文：

```text
enhanced_child =
    child_feat
    + parent_weight * gate * parent_feat
    + residual_weight * (child_feat - parent_feat)
```

其中：

- `parent_feat` 提供父物体语义上下文。
- `residual = child_feat - parent_feat` 保留部件局部差异。
- `gate` 由 child-parent cosine similarity 控制，避免强行把无关 parent 信息注入 child。

这一步解决的是特征层面的链接：**part superpoint 既保留自己的局部语义，也继承 parent object 的上下文**。

### 3.5 推理时如何从 parent 约束 part

主推理入口在：

```text
remote_thgs_patch/test_lerf.py
```

当启用 `--refine_parts` 且 `plan.role == "part"` 时，代码进入 part 专用分支：

1. 用 `target_terms` 计算主语义相似度：

```text
primary_sim = similarity_for_terms(vlm, snag.feat, target_terms)
```

例如 `bear nose` 会同时使用 `bear nose`, `nose`, `nose of bear`, `bear's nose`。

2. 用 `part_terms` 计算局部 part 相似度：

```text
aux_sim = similarity_for_terms(vlm, snag.feat, part_terms)
```

例如 `nose`, `bear nose`。这一步让候选更偏向局部部件，而不是整个 parent。

3. 组合主语义和 part 语义得到候选：

```text
score = primary_val + part_aux_weight * aux_val
```

对应函数：

```text
reranked_candidates(primary_sim, aux_sim, part_levels, part_candidate_topk, part_aux_weight)
```

4. 用 `anchor_terms` 渲染 parent anchor mask：

```text
anchor_mask = plan_anchor_mask(..., plan.anchor_terms, ...)
```

例如先渲染 `bear` 或 `stuffed bear` 的区域，作为 nose 必须落入的空间范围。

5. 用 parent anchor 过滤 part 候选：

```text
containment = area(candidate_mask ∩ anchor_mask) / area(candidate_mask)
```

如果 containment 小于 `part_anchor_min_containment`，该候选会被丢弃。保留下来的候选按下式重排：

```text
final_score =
    candidate_score
    + part_anchor_weight * containment
    - part_area_penalty * area_ratio
```

对应函数：

```text
anchor_filtered_related_gaussian(...)
```

这一步是最关键的 parent-part 链接：**part 候选必须在 parent anchor 内部或高度重合，同时面积不能过大**。

6. 最终把选中的 superpoint 转成 Gaussian mask：

```text
gaussian_from_candidates(snag, selected)
```

然后通过 THGS 原本的 Gaussian render 流程输出 2D mask。

### 3.6 Fine segment 后处理中的 parent-part 链接

除了 Gaussian 推理分支，还保留了一个更可解释的 fine segment 后处理：

```text
remote_thgs_patch/parent_part_proposal_render.py
```

它的逻辑是：

1. `anchor_candidates(prompt, labels)` 根据 `heuristic_query_plan()` 找 parent/base label。
2. `load_anchor()` 从 baseline 输出中读取 parent mask，并做轻微 dilation。
3. 遍历 fine segment map 中所有 segment。
4. 对每个 segment 计算：

```text
containment = area(segment ∩ anchor_mask) / area(segment)
cover = area(segment ∩ anchor_mask) / area(anchor_mask)
```

5. 如果 segment 不在 parent anchor 内，直接过滤：

```text
containment < min_containment -> reject
```

6. 对保留 segment 打分：

```text
score =
    containment_weight * containment
    - cover_penalty * abs(log(cover / target_cover))
    - level_penalty
    + semantic_bonus
    - same_label_area_penalty
    + consistency_bonus
```

其中：

- `semantic_bonus` 来自 segment 的 `text_label` 是否等于 `part_terms` 或 head noun。
- `target_cover` 控制 part 占 parent 的合理比例，例如 `hooves` 更小，`nose` 稍大。
- `same_label_area_penalty` 防止 text label 正确但区域过大，避免把 parent 整体选成 part。
- `consistency_bonus` 奖励跨层级重复出现、边界较稳定的 segment。

最后：

```text
out &= anchor_mask
```

保证输出 part mask 被裁剪在 parent anchor 范围内。

### 3.7 一个具体例子：`bear nose`

以 `teatime / bear nose` 为例，代码中的链路是：

```text
prompt: bear nose
  -> QueryReasoner
  -> role = part
  -> anchor_terms = [bear]
  -> part_terms = [nose, bear nose]
```

然后：

```text
anchor_terms [bear]
  -> 通过 CLIP/NAG 找 bear superpoint
  -> 渲染成 anchor_mask
```

同时：

```text
part_terms [nose, bear nose]
  -> 在 fine levels 上找 nose-like superpoints/segments
  -> 候选必须满足 containment(candidate, bear_anchor) 足够高
  -> 候选面积接近 nose 对 parent 的合理比例
  -> 输出局部 nose mask
```

所以 parent 和 part 的关系不是通过 hard-coded 类别表直接连起来，而是通过：

1. prompt 解析得到 `anchor_terms` 和 `part_terms`；
2. NAG 层级图提供 child-parent superpoint 映射；
3. parent anchor mask 提供空间约束；
4. part similarity 和面积先验提供局部选择；
5. containment 打分把二者绑定到最终 mask。

### 3.8 本轮保留和调整的 parent-part 实现

本轮没有继续使用 `scene fill` 作为主要方法，而是保留了更可解释的 parent-part 后处理与诊断脚本：

```text
remote_thgs_patch/test_lerf.py
remote_thgs_patch/query_reasoner.py
remote_thgs_patch/nag_data.py
remote_thgs_patch/parent_part_proposal_render.py
remote_thgs_patch/diagnose_parent_part_candidates.py
```

其中关键调整包括：

| 调整 | 作用 |
|---|---|
| `QueryReasoner` 将 prompt 分成 object / part / modifier 角色 | 避免所有 prompt 使用同一种路由策略 |
| `hier_soft` 层级语义特征 | 在 Gaussian 语义特征中加入 parent feature 与 residual feature |
| `refine_parts` 分支 | 只对 part 查询启用更细粒度候选重排 |
| `plan_anchor_mask` / anchor terms | 用父物体或上下文物体生成空间锚点 |
| `anchor_filtered_related_gaussian` | 候选必须与 parent anchor 有足够 containment |
| `part_area_penalty` / `target_cover` | 防止 part mask 扩张成整物体 |
| `part_diversity_iou` | 避免 top-k 候选高度重复 |
| `parent_part_proposal_render.py` | 从 fine segment map 中按 containment、cover、语义 bonus、跨层一致性选择 part mask |
| `diagnose_parent_part_candidates.py` | 输出候选排序和 GT IoU，用于判断失败是候选缺失还是排序错误 |

这部分更适合作为论文或汇报中的 **细粒度 part 修正模块 / failure analysis 工具**，而不是作为解决 object-level failure 的主方法。

### 3.9 Parent-part 与 SoM rerouting 的分工

后续引入 GPT-SoM rerouting，并不是否定 parent-part，而是因为二者解决的问题不同：

| 模块 | 主要解决对象 | 主要瓶颈 | 本轮结论 |
|---|---|---|---|
| parent-part 语义继承 | `bear nose`, `hooves`, `hand` 等 part prompt | part 容易被父物体或相邻区域淹没 | 对 part 查询有解释性和局部收益，但难以显著提升整体 object mIoU |
| GPT-SoM rerouting | `bowl`, `plate`, `sake cup`, `kamaboko`, `corn`, `onion segments` 等 object/thing prompt | 正确候选存在但语义路由选错 | 对 object 低分样例提升明显，是本轮总体指标提升的主要来源 |

因此，汇报时建议这样表述：

> Parent-part 改进验证了层级语义和空间锚点对细粒度 part 查询是必要的，但整体提升不高说明 THGS 的主要 object failure 不是 parent-part 关系缺失，而是候选选择和语义路由不足。基于这个判断，后续转向 ReLaGS-inspired SoM rerouting，用 VLM 在候选 mask 中做显式选择。

## 4. 方法调整思路

本轮没有继续依赖 scene fill，而是设计了一个 ReLaGS-inspired 的 SoM 候选选择实验。

准确表述应为：

> ReLaGS-inspired SoM-based VLM rerouting

而不是完整 ReLaGS。

原因：

- 采用了 ReLaGS 中类似的 Set-of-Mark 编号候选 + VLM 选择思想。
- 但没有构建完整 3D scene graph。
- 没有做 superpoint-level lifting。
- 在 THGS 中它是一个 post-hoc rerouting / diagnostic module。

整体流程：

1. 找出 baseline IoU `< 0.1` 的低分 prompt-frame 样例。
2. 为每个低分样例生成候选 mask。
3. 将候选 mask 画成编号 SoM 面板。
4. 使用 VLM 从编号候选中选择最符合目标语义的 mask。
5. 用选中的 mask 替换 baseline 输出。
6. 与 baseline 和 oracle candidate 上限对比。

## 5. 代码改动

新增主要脚本：

```text
remote_thgs_patch/llm_som_object_reroute.py
```

同时保留并整理了 parent-part 相关 patch：

```text
remote_thgs_patch/test_lerf.py
remote_thgs_patch/parent_part_proposal_render.py
remote_thgs_patch/diagnose_parent_part_candidates.py
remote_thgs_patch/merge_scene_fill_soft_roles.py
```

其中 `merge_scene_fill_soft_roles.py` 仅作为旧实验记录保留，不建议作为当前主方案使用。

远程同步位置：

```text
/home/Groups/group2/Working/tyy/project/THGS-main/llm_som_object_reroute.py
```

脚本支持以下功能：

| 命令 | 功能 |
|---|---|
| `build` | 构建候选、生成 SoM 编号图、保存候选分析 |
| `qwen` | 使用本地 Qwen3-VL-8B-Instruct 做候选选择 |
| `apply` | 将选择结果应用到 baseline prediction tree |
| `report` | 输出 per-target 汇总报告 |

后续新增：

- `--all_prompts`：允许对场景中所有 prompt 自动构建 SoM，而不是只限制 ramen 六个 object。
- `gpt54` apply/report 模式：用于应用 GPT-5.4 API 的选择结果。

本地 Git 提交记录：

```text
b08f319 Add LLM SoM ramen rerouting experiment
946fdff Support GPT SoM selection apply mode
dbb7207 Allow SoM build over all prompts
```

说明：GitHub push 曾失败，原因是本机 GitHub SSH publickey 权限不通过。

## 6. 候选池构建修正

最初候选主要来自：

- baseline prediction
- proposal clip outputs
- scene_opt 诊断源
- fine segment map 中与目标文本相似的 segment

但发现一个问题：

> 正确 mask 经常被 CLIP/text label 误标成其他类别，因此如果只用文本标签筛选，会漏掉真正正确的视觉候选。

例如：

| target | 正确 segment 的错误标签 |
|---|---|
| `bowl` | `plastic ladle` |
| `corn` | `egg`, `yellow pouf` |
| `plate` | `plastic ladle`, `spoon` |

因此加入了：

```text
segment_map_generic
```

即基于目标面积先验的通用 segment 候选，不再完全相信自动文本标签。

该改动显著提高候选池 oracle 上限。

以 `ramen` 低分样例为例：

| target | baseline IoU | 修正后 oracle IoU |
|---|---:|---:|
| bowl | 0.0000 | 0.9653 |
| corn | 0.0735 | 0.6147 |
| kamaboko | 0.0573 | 0.9096 |
| onion segments | 0.0000 | 0.5663 |
| plate | 0.0000 | 0.8699 |
| sake cup | 0.0284 | 0.9102 |

这说明候选池本身是有潜力的，主要瓶颈是选候选。

## 7. 本地 Qwen3-VL 实验

服务器中找到的模型：

```text
/home/Groups/group2/.cache/modelscope/hub/models/Qwen/Qwen3-VL-8B-Instruct
```

该模型实际是 `Qwen3-VL-8B-Instruct`，不是纯文本 Qwen3-8B。

环境问题：

- `thgs` 环境有 transformers，但 PyTorch 版本过低。
- 最终使用 `esam3_312` 环境的 torch/torchvision，并通过脚本兼容加载 transformers。

本地 Qwen3-VL 在 `ramen` 31 个低分样例上的结果：

| 方法 | 低分样例 IoU | ramen 全场 mIoU |
|---|---:|---:|
| baseline | 0.0294 | 0.4209 |
| Qwen3-VL SoM | 0.1721 | 0.4856 |
| oracle | 0.7849 | 0.7596 |

结论：

> 本地 Qwen3-VL 能带来一定提升，但选择不稳定，经常偏向大 mask 或第一个候选。

## 8. GPT-5.4 API 实验

用户提供 OpenAI 兼容 API 后，先验证模型列表，确认可用模型包括：

```text
gpt-5.4
gpt-5.4-mini
gpt-5.5
```

API key 没有写入仓库、脚本或文档，只以一次性环境变量使用。

由于需要上传 SoM 面板图片到外部 API，已在用户明确同意后继续实验。

先在 `ramen` 31 个低分样例上测试 GPT-5.4：

| 方法 | 低分样例 IoU | ramen 全场 mIoU |
|---|---:|---:|
| baseline | 0.0294 | 0.4209 |
| Qwen3-VL SoM | 0.1721 | 0.4856 |
| GPT-5.4 SoM | 0.5679 | 0.6633 |
| oracle | 0.7849 | 0.7596 |

GPT-5.4 明显优于本地 Qwen3-VL。

## 9. 四场景扩展实验

扩展到 LERF-OVS 四个场景：

```text
figurines
ramen
teatime
waldo_kitchen
```

处理对象：

> 所有 baseline IoU `< 0.1` 的 prompt-frame 样例。

低分样例统计：

| Scene | 低分样例数 |
|---|---:|
| figurines | 11 |
| ramen | 33 |
| teatime | 4 |
| waldo_kitchen | 6 |
| total | 54 |

四场景总体结果：

| 方法 | Overall mIoU | mAcc | mP | mR | F1 |
|---|---:|---:|---:|---:|---:|
| baseline THGS | 0.5885 | 0.9776 | 0.6674 | 0.7141 | 0.6521 |
| GPT-5.4 SoM rerouting | 0.6748 | 0.9855 | 0.7662 | 0.7685 | 0.7449 |
| oracle 候选上限 | 0.7362 | 0.9882 | 0.8294 | 0.8289 | 0.8069 |

分场景 mIoU：

| Scene | baseline | GPT-5.4 SoM | oracle |
|---|---:|---:|---:|
| figurines | 0.5491 | 0.5982 | 0.6436 |
| ramen | 0.4209 | 0.6226 | 0.7778 |
| teatime | 0.8186 | 0.8325 | 0.8379 |
| waldo_kitchen | 0.5656 | 0.6460 | 0.6854 |

结论：

> GPT-5.4 SoM rerouting 将四场景总体 mIoU 从 0.5885 提升到 0.6748，提升 +0.0863。oracle 上限为 0.7362，说明候选池仍有进一步利用空间。

## 10. 可视化结果、对比解释与保存路径

本节建议作为给老师汇报时的重点页。每个可视化样例都按同一逻辑解释：

```text
baseline 失败现象
-> 为什么原 THGS 路由容易失败
-> 本轮改动如何介入
-> 可视化中应该看什么
-> 指标或 oracle 上限说明是否有效
```

### 10.1 四场景 SoM 编号候选图：证明“正确候选是否存在”

根目录：

```text
/home/Groups/group2/Working/tyy/project/THGS-main/output/gpt54_som_low_all_artifacts/
```

每个场景下：

```text
/home/Groups/group2/Working/tyy/project/THGS-main/output/gpt54_som_low_all_artifacts/<scene>/som_panels/
```

示例：

```text
/home/Groups/group2/Working/tyy/project/THGS-main/output/gpt54_som_low_all_artifacts/ramen/som_panels/ramen/frame_00006/bowl_som.png
```

该图展示：

- 原图
- 编号候选 mask
- 每个候选的 overlay
- 候选来源和面积

本地 GitHub 复现包建议保存代表性图到：

```text
visualizations/som_panels/
```

汇报中建议至少展示以下六个 `ramen` object 低分样例：

| 样例 | baseline 问题 | 看 SoM 图时要说明什么 | 结论 |
|---|---|---|---|
| `bowl` | baseline 为空 mask | SoM 候选中存在完整 bowl，但文本路由把它误标到其他类别 | 候选存在，失败来自候选选择 |
| `plate` | baseline 为空或选到局部 | plate 候选常被 `plastic ladle/spoon` 等标签污染 | 不能只按自动 text label 筛候选 |
| `sake cup` | 小 object 漏检 | SoM 候选有小杯子，但原路由置信度低 | VLM 视觉选择比 CLIP label 更稳 |
| `kamaboko` | 大面积误检 | 候选中有更贴合的局部鱼糕区域 | rerouting 能抑制过大 mask |
| `corn` | 易选到黄色相似区域 | 候选中存在 corn-like 区域，但与 egg/yellow region 混淆 | 仍依赖候选质量和 VLM 判别 |
| `onion segments` | 细小目标选错 | 候选很碎，oracle 也低于大物体 | 小碎片仍是后续难点 |

图片引用模板：

```markdown
![ramen bowl SoM](visualizations/som_panels/ramen_bowl_frame_00006_som.png)
```

已保存到 GitHub 的代表性 SoM 图：

| 目标 | 本地可视化文件 | 作用 |
|---|---|---|
| `bowl` | `visualizations/som_panels/ramen_bowl_frame_00006_som.png` | 展示正确 bowl 候选存在 |
| `plate` | `visualizations/som_panels/ramen_plate_frame_00006_som.png` | 展示 plate 候选被文本标签污染时仍可视觉选择 |
| `sake cup` | `visualizations/som_panels/ramen_sake_cup_frame_00006_som.png` | 展示小 object 候选 |
| `kamaboko` | `visualizations/som_panels/ramen_kamaboko_frame_00006_som.png` | 展示局部鱼糕候选与过大候选的差异 |
| `corn` | `visualizations/som_panels/ramen_corn_frame_00024_som.png` | 展示黄色相似区域混淆 |
| `onion segments` | `visualizations/som_panels/ramen_onion_segments_frame_00006_som.png` | 展示碎片状小目标候选 |

![ramen bowl SoM](visualizations/som_panels/ramen_bowl_frame_00006_som.png)

### 10.2 GPT-5.4 before/after 对比图：证明“改后是否有效”

根目录：

```text
/home/Groups/group2/Working/tyy/project/THGS-main/output/gpt54_som_low_all_artifacts/<scene>/applied_compare/gpt54/
```

示例：

```text
/home/Groups/group2/Working/tyy/project/THGS-main/output/gpt54_som_low_all_artifacts/ramen/applied_compare/gpt54/ramen/frame_00006/bowl.png
```

对比图包含：

- 原图
- GT
- baseline mask
- GPT-5.4 rerouting 后的 mask

本地 GitHub 复现包建议保存代表性图到：

```text
visualizations/gpt54_compare/
```

汇报时不要只说“mIoU 提升”，需要逐图说明：

| 对比项 | 应该观察的视觉差异 | 说明 |
|---|---|---|
| 原图 vs GT | 目标真实位置和大小 | 明确目标不是语义歧义 |
| baseline vs GT | 原 THGS 是空、过大还是错选 | 定性失败类型 |
| GPT-5.4 rerouting vs GT | rerouting 是否选中更接近 GT 的候选 | 证明 VLM 选择有效 |
| GPT-5.4 vs oracle | 还差在哪里 | 判断后续该改 VLM 选择还是候选池 |

图片引用模板：

```markdown
![ramen bowl GPT compare](visualizations/gpt54_compare/ramen_bowl_frame_00006.png)
```

已保存到 GitHub 的 GPT-5.4 before/after 对比图：

| 目标 | 本地可视化文件 | 汇报时说明 |
|---|---|---|
| `bowl` | `visualizations/gpt54_compare/ramen_bowl_frame_00006.png` | baseline 为空，GPT-SoM 选中 bowl 候选 |
| `plate` | `visualizations/gpt54_compare/ramen_plate_frame_00006.png` | 从错误或空预测转为更贴近 plate 的候选 |
| `sake cup` | `visualizations/gpt54_compare/ramen_sake_cup_frame_00006.png` | 小 object 漏检被部分修正 |
| `kamaboko` | `visualizations/gpt54_compare/ramen_kamaboko_frame_00006.png` | 抑制过大误检，选择更局部候选 |
| `corn` | `visualizations/gpt54_compare/ramen_corn_frame_00024.png` | 展示 GPT 对黄色细粒度目标的选择效果和残余混淆 |
| `onion segments` | `visualizations/gpt54_compare/ramen_onion_segments_frame_00006.png` | 展示碎片目标仍是困难样例 |

![ramen bowl GPT compare](visualizations/gpt54_compare/ramen_bowl_frame_00006.png)

### 10.3 oracle 上限对比图：证明“候选池上限”

```text
/home/Groups/group2/Working/tyy/project/THGS-main/output/gpt54_som_low_all_artifacts/<scene>/applied_compare/oracle/
```

oracle 图不是最终方法结果，而是诊断候选池的 upper bound。解释时要强调：

- 如果 oracle 高、GPT 低：候选存在，VLM 选择还不够稳。
- 如果 oracle 也低：候选池本身缺目标或边界质量差。
- 本轮四场景 oracle mIoU `0.7362`，GPT-5.4 mIoU `0.6748`，说明大部分提升空间来自 candidate selection，但候选池仍可继续优化。

本地 GitHub 复现包建议保存代表性图到：

```text
visualizations/oracle_compare/
```

已保存代表性 oracle 图：

```text
visualizations/oracle_compare/ramen_bowl_frame_00006_oracle.png
```

![ramen bowl oracle](visualizations/oracle_compare/ramen_bowl_frame_00006_oracle.png)

### 10.4 Parent-part 可视化：证明“parent 是否真的约束 part”

parent-part 不应只展示最终指标，必须展示 part 是否落在 parent 内。建议展示 `teatime / bear nose` 和 `teatime / hooves`：

| 样例 | baseline 问题 | parent-part 观察点 | 是否有效 |
|---|---|---|---|
| `bear nose` | part 容易扩散到 bear 以外或选到整脸/整物体 | 输出 mask 是否被限制在 bear anchor 内，且面积接近 nose | 若 containment 高且面积小，说明 parent anchor 生效 |
| `hooves` | 易选到整只羊/玩偶或其它小黑区域 | 输出是否只覆盖 parent 下方小部件 | 若 mask 不再扩成 parent object，说明面积先验有效 |

对应路径：

```text
/home/Groups/group2/Working/tyy/project/THGS-main/output/render_parent_part/lerf
/home/Groups/group2/Working/tyy/project/THGS-main/output/parent_part_debug.json
```

本地 GitHub 复现包建议保存代表性图到：

```text
visualizations/parent_part/
```

已保存到 GitHub 的 parent-part 代表性图：

| 样例 | 本地可视化文件 | 汇报时说明 |
|---|---|---|
| `bear nose` SoM | `visualizations/parent_part/teatime_bear_nose_frame_00002_som.png` | 展示 part 候选与 parent anchor 的关系 |
| `bear nose` 对比 | `visualizations/parent_part/teatime_bear_nose_frame_00002_gpt54.png` | 展示 baseline / GT / rerouting 结果 |
| `hooves` SoM | `visualizations/parent_part/teatime_hooves_frame_00025_som.png` | 展示小 part 候选容易和 parent/其它小区域混淆 |
| `hooves` 对比 | `visualizations/parent_part/teatime_hooves_frame_00025_gpt54.png` | 展示 parent 约束和面积先验是否抑制整物体误选 |

![teatime bear nose SoM](visualizations/parent_part/teatime_bear_nose_frame_00002_som.png)

![teatime bear nose compare](visualizations/parent_part/teatime_bear_nose_frame_00002_gpt54.png)

![teatime hooves SoM](visualizations/parent_part/teatime_hooves_frame_00025_som.png)

![teatime hooves compare](visualizations/parent_part/teatime_hooves_frame_00025_gpt54.png)

解释 parent-part 图时要结合 `parent_part_debug.json` 中的字段：

| debug 字段 | 含义 | 汇报时怎么解释 |
|---|---|---|
| `anchors` | 当前 part 使用了哪些 parent/base mask | 说明 parent 来源 |
| `containment` | part candidate 有多少比例落在 parent 内 | 证明空间约束是否生效 |
| `cover` | part 占 parent 的比例 | 判断是否选成整物体 |
| `text_label` | segment 自动语义标签 | 判断 text label 是否可靠 |
| `score` | 综合 containment、面积、语义 bonus 后的分数 | 说明为什么选这个候选 |

### 10.5 最终预测输出

GPT-5.4 rerouting 后预测：

```text
/home/Groups/group2/Working/tyy/project/THGS-main/output/render_gpt54_som_low_all/lerf
```

oracle 上限预测：

```text
/home/Groups/group2/Working/tyy/project/THGS-main/output/render_gpt54_som_low_all_oracle/lerf
```

### 10.6 Parent-part 诊断结果

parent-part 候选诊断脚本：

```text
/home/Groups/group2/Working/tyy/project/THGS-main/diagnose_parent_part_candidates.py
```

parent-part 后处理输出：

```text
/home/Groups/group2/Working/tyy/project/THGS-main/output/render_parent_part/lerf
```

debug JSON：

```text
/home/Groups/group2/Working/tyy/project/THGS-main/output/parent_part_debug.json
```

该 JSON 记录每个 part prompt 使用的 anchors、被选中的 segment layer / segment_id、面积、containment、cover、text label 和最终 score，便于后续复盘 `bear nose`、`hooves` 等 part 查询是否真正受 parent 约束。

## 11. 适合写进论文的表述

不建议直接写成：

> We use ReLaGS.

更准确的写法：

> ReLaGS-inspired SoM-based VLM rerouting.

或：

> Inspired by the Set-of-Mark annotation strategy in ReLaGS, we construct numbered candidate masks and query a VLM to reroute low-confidence open-vocabulary predictions.

建议论文中定位为：

1. failure analysis
2. diagnostic experiment
3. optional VLM-assisted rerouting module
4. upper-bound study for candidate selection

核心论点：

> THGS 的大目标 object failure 很多不是因为没有视觉候选，而是因为语义路由没有从候选中选出正确 mask。GPT-5.4 SoM rerouting 显著缩小了 baseline 与 oracle candidate 上限之间的差距。

## 12. 后续建议

当前 GPT-5.4 API 结果很强，但作为论文主方法仍有风险：

- 闭源模型不可完全复现。
- API 成本和稳定性可能被审稿人质疑。
- 依赖外部模型，不一定适合做核心 contribution。

更稳妥的后续方向：

1. 使用 GPT-5.4 结果作为 pseudo-label，训练一个 lightweight reranker。
2. 将 SoM rerouting 蒸馏到开源 VLM 或 CLIP-based reranker。
3. 只在论文中把 GPT-5.4 作为 diagnostic / upper-bound evidence。
4. 进一步改进候选池，缩小 GPT 与 oracle 的差距。
5. 对失败类如 `corn`, `onion segments`, `figurines` 中的小物体做更细粒度候选生成。

## 13. 汇报用一句话总结

本轮实验表明，parent-part 语义继承和空间锚点能改善部分细粒度 part 查询，但它不是 object-level 指标提升有限的主要原因。THGS 在 object-level 上的核心瓶颈是正确候选存在但语义路由没有选中。通过 ReLaGS-inspired GPT-SoM rerouting，四场景总体 mIoU 从 0.5885 提升到 0.6748，oracle 上限达到 0.7362，证明引入视觉大模型进行候选选择是有效方向，也为后续设计可复现的 reranker 或蒸馏模块提供了明确依据。
