# THGS 层级语义与 SoM Rerouting 改进汇报

日期：2026-06-09  
方向：开放词汇 3D Gaussian Splatting 语义分割  
数据集：LERF-OVS 四场景

## 1. 汇报目的

本次汇报主要复盘我对 THGS 细粒度开放词汇分割结果的分析和改进过程。我的重点不是简单报告“指标提升了多少”，而是说明：

1. THGS 原始失败主要发生在哪里；
2. parent-part 语义约束为什么有必要、在哪些样例有效、在哪些样例失败；
3. 为什么后续需要转向 SoM rerouting；
4. SoM rerouting 是否真正验证了“候选存在但选择错误”这个判断；
5. 目前结果能支持什么结论，哪些地方还需要继续改。

整体结论是：

> THGS 的 part-level 问题和 object-level 问题并不完全相同。parent-part 约束能改善一部分“part 被 parent object 淹没”的细粒度样例，但不能解释 object 类整体提升不高的问题。进一步的 SoM rerouting 实验表明，很多 object failure 并不是没有视觉候选，而是语义路由没有从候选中选出正确 mask。

## 2. 初始观察：THGS 的失败不是单一问题

我首先检查 baseline 和前期改进结果，发现整体提升不明显的原因并不是单一模块失效，而是至少包含两类问题。

第一类是 **part query 被 parent object 淹没**。例如 `bear nose` 这类查询，本来目标是局部鼻子，但模型容易选成整只熊。这类问题需要 parent-part 语义和空间约束。

第二类是 **object query 候选存在但语义选择错误**。例如 `bowl`、`plate`、`sake cup` 这类目标，baseline 经常为空或选错区域，但在候选池中其实能找到接近正确目标的 mask。这类问题更像 candidate selection / rerouting，而不是 parent-part 问题。

典型失败如下：

| 失败类型 | 代表样例 | 我的判断 |
|---|---|---|
| part 扩散成 parent object | `bear nose` | 需要 parent anchor 和 part 面积约束 |
| 小 part 候选不稳定 | `hooves` | parent-part 也可能失败，需要看候选质量 |
| object 空 mask | `bowl`, `plate`, `sake cup` | 正确候选可能存在，但原路由没选中 |
| 大面积误检 | `kamaboko`, `corn` | 需要在多个候选中选择更合理区域 |
| 碎片目标困难 | `onion segments` | 候选边界弱，oracle 上限也有限 |

因此，我后续把 parent-part 和 SoM 分开评价，而不是混成一个方法。

## 3. Parent-part：为什么做、怎么判断有效

### 3.1 方法动机

parent-part 的出发点是：当文本查询是一个 part 时，模型不应该在整幅图中自由寻找相似区域，而应该先确定这个 part 属于哪个 parent object，再在 parent 内部寻找局部部件。

例如：

```text
bear nose = parent: bear + part: nose
hooves = parent: sheep / stuffed animal + part: hooves
```

我的目标是让 part mask 满足两个条件：

1. 它应该落在 parent object 附近或内部；
2. 它的面积应该像一个 part，而不是扩张成整个 parent object。

### 3.2 成功案例：bear nose

`bear nose` 是 parent-part 约束有效的代表样例。baseline 把 `bear nose` 几乎选成整只熊，这说明原始 THGS 对 part query 的处理会被 parent object 的强语义吸引。

加入 parent-part 约束后，mask 从整只熊收缩到了鼻子局部。

| 方法 | IoU | mask 面积 | 解释 |
|---|---:|---:|---|
| baseline | 0.0642 | 154099 | 几乎选成整只熊 |
| parent-part | 0.9840 | 9761 | 基本贴合鼻子区域 |
| GT | - | 9906 | 目标确实是局部 part |

![parent-part bear nose success](visualizations/parent_part/teatime_frame_00002_bear_nose_parent_part_compare.png)

这个案例说明：当 parent anchor 正确、局部候选存在时，parent-part 能有效抑制 part 向 parent object 扩散。

### 3.3 失败案例：hooves

`hooves` 是 parent-part 失败的代表样例。这个目标更小、更不显著，候选本身不稳定。baseline 没有选到正确蹄子，parent-part 后处理也没有改善，反而选到了白羊身体或头部附近较大的区域。

| 方法 | IoU | mask 面积 | 解释 |
|---|---:|---:|---|
| baseline | 0.0000 | 1727 | 没有覆盖正确蹄子 |
| parent-part | 0.0000 | 18302 | 选到错误的大区域 |
| GT | - | 4678 | 目标是更小的局部 part |

![parent-part hooves failure](visualizations/parent_part/teatime_frame_00025_hooves_parent_part_compare.png)

这个失败说明 parent-part 不是万能的。它依赖两个前提：parent 定位要可靠，候选池中要有合理的 part segment。如果候选池没有稳定给出真正的蹄子区域，parent-part 的面积约束和空间约束也无法凭空生成正确 mask。

### 3.4 parent-part 的阶段性结论

我对 parent-part 的结论是谨慎的：

> parent-part 对 `bear nose` 这类 parent 明确、part 候选存在的查询有效；但对 `hooves` 这类小、低显著性、候选不稳定的部件效果不稳定。因此它适合作为细粒度 part 修正模块，但不能作为解决整体 object-level failure 的主方法。

这也解释了为什么前期只做 parent-part 时，整体指标提升并不明显：LERF 四场景中的大部分低分样例并不是 part-parent 关系问题，而是 object 候选选择问题。

## 4. 为什么转向 SoM Rerouting

在检查 `ramen` 场景时，我发现很多 object 类目标的 baseline 非常低，甚至为空 mask，但候选池中可以看到接近正确目标的 mask。

这让我把问题从“如何直接生成一个 mask”转为：

> 如果正确 mask 已经在候选池里，是否可以让视觉大模型从编号候选中选出来？

这就是 SoM rerouting 的动机。它不是完整复现 ReLaGS，而是借鉴 Set-of-Mark 的思路，把候选 mask 编号后交给视觉语言模型选择。

## 5. SoM 候选证据：正确候选确实存在

以 `ramen / bowl` 为例，baseline 的 bowl 预测很差，但 SoM 候选图中可以看到接近 bowl 的候选。这说明失败不完全是候选缺失，而是原始语义路由没有选到正确候选。

![ramen bowl SoM](visualizations/som_panels/ramen_bowl_frame_00006_som.png)

我进一步检查了多个 object 低分样例：

| 目标 | baseline 问题 | SoM 候选观察 | 说明 |
|---|---|---|---|
| `bowl` | 空 mask | 有完整 bowl 候选 | 候选存在，选择失败 |
| `plate` | 空或局部错误 | plate 候选被相似标签干扰 | 不能只相信自动文本标签 |
| `sake cup` | 小 object 漏检 | 有小杯子候选 | 需要视觉判断 |
| `kamaboko` | 大面积误检 | 有更局部的鱼糕候选 | rerouting 可抑制过大区域 |
| `corn` | 黄色区域混淆 | corn 与 egg 等区域相似 | 仍有选择难度 |
| `onion segments` | 细碎目标选错 | 候选碎片化明显 | 小碎片仍是难点 |

## 6. GPT-SoM 是否有效

在 `ramen / bowl` 的 before-after 对比中，可以看到 baseline 与 GT 偏差明显，而 GPT-SoM rerouting 后的 mask 更接近目标。

![ramen bowl GPT compare](visualizations/gpt54_compare/ramen_bowl_frame_00006.png)

这个现象在多个 object 上都有体现：

| 目标 | 观察结果 | 结论 |
|---|---|---|
| `bowl` | 从空/错 mask 转为更接近 bowl 的候选 | 明显有效 |
| `plate` | 能从相似候选中选出更接近 plate 的区域 | 有效 |
| `sake cup` | 小 object 漏检有所缓解 | 有效但仍受候选质量限制 |
| `kamaboko` | 能抑制过大误检 | 有效 |
| `corn` | 仍有黄色相似区域干扰 | 部分有效 |
| `onion segments` | 碎片候选不稳定 | 仍是困难样例 |

为了判断候选池上限，我还比较了 oracle candidate。oracle 不是实际方法，而是用来回答“如果候选选择完全正确，最多能到什么水平”。

![ramen bowl oracle](visualizations/oracle_compare/ramen_bowl_frame_00006_oracle.png)

## 7. 指标结果

### 7.1 ramen 低分样例

先在 `ramen` 的低分 object 样例上测试：

| 方法 | 低分样例 IoU | ramen 全场 mIoU |
|---|---:|---:|
| baseline | 0.0294 | 0.4209 |
| Qwen3-VL SoM | 0.1721 | 0.4856 |
| GPT-5.4 SoM | 0.5679 | 0.6633 |
| oracle | 0.7849 | 0.7596 |

这个结果说明：

- Qwen3-VL 有提升，但选择不稳定；
- GPT-5.4 明显更强；
- oracle 仍高于 GPT-5.4，说明候选池中还有更多可利用空间。

### 7.2 四场景总体结果

扩展到 LERF-OVS 四个场景后，整体结果如下：

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

可以看到，GPT-SoM 将整体 mIoU 从 `0.5885` 提升到 `0.6748`，提升 `+0.0863`。oracle 上限为 `0.7362`，说明主要瓶颈确实在候选选择，但候选池和选择器仍没有完全用尽潜力。

## 8. SoM 在 part query 上的成功与失败

为了避免和 parent-part 混淆，我单独看了 SoM 对 part query 的表现。这里的图只说明 SoM 的候选选择能力，不用于证明 parent-part 模块本身。

`bear nose` 是 SoM 成功案例：候选中存在正确局部 nose，GPT-SoM 能从整熊错误预测中纠正到局部 part。

![teatime bear nose SoM](visualizations/som_part_cases/teatime_bear_nose_frame_00002_som.png)

![teatime bear nose compare](visualizations/som_part_cases/teatime_bear_nose_frame_00002_gpt54.png)

`hooves` 是 SoM 失败案例：候选图中有大量人腿、桌椅、盘子等干扰区域，正确蹄子候选不明显，因此 GPT-SoM 仍选错。

![teatime hooves SoM](visualizations/som_part_cases/teatime_hooves_frame_00025_som.png)

![teatime hooves compare](visualizations/som_part_cases/teatime_hooves_frame_00025_gpt54.png)

这说明 SoM 也依赖候选池质量。如果正确候选不明显或不存在，视觉语言模型也难以稳定选择正确目标。

## 9. 我目前认为可以支持的结论

根据以上实验和可视化，我认为目前可以支持以下结论：

1. THGS 的失败可以分为 part-level 和 object-level 两类，不能用一个模块解释全部问题。
2. parent-part 对部分 part query 有效，尤其是 `bear nose` 这种 parent 明确、局部候选存在的情况。
3. parent-part 对 `hooves` 等小部件不稳定，说明它依赖候选质量和 parent anchor 的可靠性。
4. object-level 的大幅低分主要来自候选选择失败，而不是候选完全缺失。
5. GPT-SoM rerouting 能显著改善 object-level 低分样例，四场景总体 mIoU 从 `0.5885` 提升到 `0.6748`。
6. oracle 上限 `0.7362` 说明还有剩余提升空间，后续应该继续改候选池和可复现 reranker。

## 10. 论文表述建议

我不建议把这部分直接写成“使用 ReLaGS”。更准确的表述应该是：

> Inspired by Set-of-Mark style visual prompting, I construct numbered candidate masks and use a vision-language model to reroute low-confidence open-vocabulary predictions.

在论文里，它更适合作为：

1. failure analysis；
2. candidate selection 上限验证；
3. VLM-assisted rerouting diagnostic；
4. 后续可蒸馏 reranker 的依据。

如果要作为主方法，还需要把闭源 GPT-5.4 的选择能力蒸馏到可复现的开源 reranker 或 CLIP-based reranker 上。

## 11. 后续工作

下一步我建议从三个方向继续：

1. **蒸馏 GPT-SoM 选择结果**：把 GPT-5.4 的选择作为 pseudo-label，训练一个轻量 reranker，降低对闭源模型的依赖。
2. **改进小 part 候选池**：重点处理 `hooves`、`onion segments` 这类小、碎、低显著性目标。
3. **将 parent-part 与 object rerouting 分开消融**：分别报告 parent-part 对 part query 的收益，以及 SoM rerouting 对 object query 的收益，避免方法贡献混在一起。

## 12. 汇报总结

本轮实验让我明确了一个关键点：THGS 的整体提升不高，并不是因为一个简单模块没有调好，而是因为 part-level 和 object-level 的失败机制不同。

parent-part 能解决一部分 part 被 parent object 淹没的问题，例如 `bear nose` 从 `0.0642` 提升到 `0.9840`；但它对 `hooves` 这类小部件仍然失败，说明该方向有明确边界。

SoM rerouting 则说明很多 object failure 的正确候选其实已经存在，只是原始语义路由没有选中。GPT-5.4 SoM rerouting 将四场景总体 mIoU 从 `0.5885` 提升到 `0.6748`，oracle 上限达到 `0.7362`，证明“候选选择”是 THGS 后续改进的重要方向。

因此，我认为后续最值得推进的是：保留 parent-part 作为细粒度 part 约束模块，同时将 GPT-SoM 的诊断结果蒸馏成一个可复现、可训练的候选 reranker。
