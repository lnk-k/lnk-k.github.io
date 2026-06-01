+++
title = 'Batch Registration Qc'
date = '2026-06-01T20:08:43+08:00'
draft = false
description = ''
tags = ['期相配准','DICOM','QC','nnU-Net']
categories = ['医学图像识别']
+++

# 多期相 CT 批量配准和 QC 记录

---

# 0. 背景

之前已经写过单个 fixed 和 moving 的配准脚本，但真正开始处理病例的时候才发现，单个脚本只能证明“这件事情理论上可以做”，不能证明“这件事情可以批量稳定地做”。

因为一个病例里面往往不止两个序列，可能有平扫、动脉期、门静脉期、延迟期，也可能夹杂 localizer、lung、std、薄层、厚层之类的序列。更麻烦的是，即使它们都来自同一个病例，也不能默认它们的切片编号就是一一对应的。

所以这次想做的是：

> 批量读取同一个病例下的多个 DICOM series，自动筛选、配准、评价配准效果，并输出可以作为 nnU-Net 多通道输入的 NIfTI 文件。

完整代码见 GitHub：[medical-image-registration-scripts](https://github.com/lnk-k/medical-image-registration-scripts)

---

# 1. 一开始的想法

最初的思路其实很朴素：

1. 读取一个病例下面所有 `seriesXXX` 文件夹；
2. 自动选一个序列作为 fixed image；
3. 其余序列作为 moving image；
4. 先通过 DICOM header 做第一轮筛选；
5. 对筛选通过的 moving 做刚性配准和重采样；
6. 配准后计算一组无标签 QC 指标；
7. 根据 QC 结果打 `PASS`、`WARNING`、`FAIL`；
8. 只把合格的序列放入最终多通道；
9. 最后对保留下来的通道做共同区域裁剪；
10. 输出 nnU-Net 的 `imagesTr` 格式。

看起来非常顺理成章，甚至有点过于顺理成章。

但是医学图像处理很多时候就是这样：流程图画出来的时候像一条笔直的路，真正跑起来的时候才发现中间到处是坑。

---

# 2. fixed image 的选择

脚本里暂时采用了一个比较直接的规则：

> 选择切片数最多的 `seriesXXX` 作为 fixed image。

这样做的原因是，切片数最多的序列通常扫描范围更完整，更适合作为统一空间的参考。

当然这不是一个绝对正确的规则。更严谨的做法应该结合：

- 序列描述；
- 扫描范围；
- 图像质量；
- 老师指定的标准期相；
- 是否为薄层或厚层重建；
- 是否包含目标器官区域。

但在当前阶段，为了先把批量流程跑通，先用“切片数最多”作为自动选择规则。

---

# 3. 配准前 header 筛选

在真正做配准之前，脚本会先比较 fixed 和 moving 的空间信息。

主要检查：

- 切片数量是否太少；
- z 轴扫描范围是否有足够重叠；
- spacing 是否差异过大；
- direction 是否差异过大；
- origin 距离是否明显异常。

其中比较重要的是 z 轴重叠率。

因为如果两个序列的扫描范围几乎没有重叠，那后面再怎么配准都没有意义。比如一个序列扫的是胸部，一个序列扫的是腹部，强行把它们做 rigid registration，只会得到一个看起来很努力但实际上毫无意义的结果。

这一步的输出是：

```text
logs/CASE001_header_compare.csv
```

小样例中，`series004` 就是在这一步被筛掉的。它和 fixed 的 z 轴重叠比例太低，所以直接标记为：

```text
FAIL: z_overlap_too_low
```

这个结果也提醒我，配准前筛选是很有必要的。否则后面的 QC 会被很多本来就不该进入配准的序列污染。

---

# 4. 第一次尝试：刚性配准后全部过不了

一开始我想得比较简单：

> 既然要配准，那就对每个 moving 都做 rigid registration，然后看配准后的结果。

于是流程是：

```text
moving image
    ↓
rigid registration
    ↓
resample to fixed space
    ↓
compute QC metrics
    ↓
decide PASS / WARNING / FAIL
```

结果跑完之后，发现一个很尴尬的问题：

> 没有任何 moving 能作为最终多通道保留下来。

检查 `qc_results.csv` 后发现，原因不是 header 过不了，而是配准后的无标签 QC 指标变差了。

具体表现为：

- MI 下降；
- NCC 下降；
- body Dice 下降；
- 有的序列质心距离变大；
- 最终被判定为不适合进入多通道。

也就是说，rigid registration 不但没有把图像配得更好，反而把本来差不多已经对齐的图像配歪了。

这个地方我才意识到一个之前忽略的问题：

> 配准算法给出的 transform 不一定比原始空间关系更可信。

如果两个序列本来 header 空间已经比较接近，强行再做一次刚性配准，优化器可能会被不同期相的强化差异、局部灰度变化或采样误差带偏。

尤其多期相 CT 不是完全相同灰度分布的图像。动脉期和门静脉期的血管、器官强化不一样，互信息虽然比单纯灰度差更稳，但它也不是万无一失。

---

# 5. 第二次尝试：identity 和 rigid 都算一遍

后来把流程改成：

> 对同一个 moving，同时计算 identity resample 和 rigid registration 的结果，然后比较两者的 QC 指标。

也就是说，不再默认 rigid 一定更好，而是让它和“不配准，只按 header 重采样”的结果竞争。

新的流程变成：

```text
moving image
    ↓
identity resample to fixed space
    ↓
compute identity QC

moving image
    ↓
rigid registration + resample
    ↓
compute rigid QC

identity QC vs rigid QC
    ↓
choose better result
```

判断逻辑大概是：

1. 如果 rigid 后 body Dice 明显变差，就不用 rigid；
2. 如果 rigid 后共同区域比例明显变差，就不用 rigid；
3. 如果 rigid 后 MI 和 NCC 都变差，也不用 rigid；
4. 否则才接受 rigid。

这样做之后，之前因为 rigid 被配歪的序列，就可以保留下来了。

在小样例中，`series005` 和 `series006` 最后都选择了：

```text
used_transform = identity
transform_decision_reason = rigid_body_dice_worse
```

也就是说，这两个序列并不是靠 rigid 通过的，而是靠“承认 rigid 没有变好”通过的。

这个点还挺关键的。因为它说明批量化流程不能有一种盲目的算法崇拜：不是只要做了 registration 就一定更科学，有时候最好的选择就是不要动它。

---

# 6. 无标签 QC 指标

因为当前没有人工标注的配准真值，所以只能先做无标签 QC。

脚本里用了几类指标：

## 6.1 header 是否一致

配准和重采样后，moving 应该已经进入 fixed 的空间网格。

因此需要检查：

- size 是否一致；
- spacing 是否一致；
- origin 是否一致；
- direction 是否一致。

如果这一步不一致，说明结果根本不能作为 nnU-Net 多通道输入，直接 `FAIL`。

## 6.2 body Dice

用一个很粗的阈值：

```python
image > -500
```

把 CT 中大致的人体区域提出来，然后计算 fixed 和 moving 的 body mask Dice。

这个指标不要求器官精细对齐，只是先判断两个图像的身体区域是不是大体重合。

## 6.3 common ratio

common ratio 用来衡量 fixed 和 moving 的共同覆盖比例。

因为不同期相扫描范围可能不同，所以只看 Dice 有时不够直观，还需要知道：

- moving 覆盖了 fixed 的多少；
- fixed 和 moving 的共同区域是否足够大；
- 最终裁剪后还能不能形成有意义的多通道。

## 6.4 centroid distance

质心距离用来粗略判断两个 body mask 的中心偏差。

但是这次我把它调整成：

> 质心距离只作为 warning，不直接作为 fail。

原因是质心本来就是一个很粗的指标。不同序列扫描范围不完全一致时，质心偏差可能变大，但图像仍然有足够大的共同区域可以用于训练。

所以它更适合作为提醒，而不是一票否决。

## 6.5 MI 和 NCC

MI 和 NCC 主要用来比较 identity 和 rigid 谁更好。

这里还有一个小坑：

如果最终选择的是 identity，那么它本来就不存在“相对 identity 的提升”。所以这种情况下不应该因为 `mi_improvement` 等于空或 0 就给它 warning。

因此后来的逻辑是：

- 如果选择 `identity`，不检查 MI/NCC 是否提升；
- 如果选择 `rigid`，才检查 rigid 相比 identity 有没有提升。

这也是第二次修改后比较重要的一点。

---

# 7. QC 状态设计

脚本最后会给每个 moving 一个状态：

```text
PASS
WARNING
FAIL
```

当前规则大致是：

- header 不一致：`FAIL`
- body Dice 太低：`FAIL`
- common ratio 太低：`FAIL`
- body Dice 或 common ratio 较低但没到失败线：`WARNING`
- 质心距离偏大：`WARNING`
- rigid 没有带来 MI/NCC 提升：`WARNING`

但是默认情况下：

> `PASS` 和 `WARNING` 都会进入最终多通道，只有 `FAIL` 会被排除。

原因是当前 QC 还处在比较保守的探索阶段。如果只保留 `PASS`，可能会误杀很多其实可用的序列。`WARNING` 更适合表示“需要人工复查”，而不是“绝对不能用”。

如果后面想更严格，可以加参数：

```bash
python scripts/batch_registration_qc.py \
  --case-dir data/CASE001 \
  --output-dir registration_qc_output \
  --case-id CASE001 \
  --only-keep-qc-pass
```

---

# 8. 自动生成 overlay 图

只看 CSV 还是太抽象，所以脚本会给每个 moving 生成一张 QC PNG。

里面包含四个视图：

```text
fixed
moving selected
overlay R=fixed G=moving
abs diff
```

其中 overlay 的规则是：

- 红色通道：fixed
- 绿色通道：moving

如果两者重合得比较好，就会接近黄色；如果红绿分离明显，就说明对应位置可能没对齐。

这个图的意义不是替代 ITK-SNAP 或 3D Slicer，而是做批量快速预览。真正要确认一个病例能不能进训练，还是应该打开三维图像逐层检查。

---

# 9. 共同有效区域裁剪

当 fixed 和保留下来的 moving 都确定之后，脚本会对这些通道求共同 body 区域。

大致逻辑是：

```text
fixed body mask
moving body mask 1
moving body mask 2
...
    ↓
取共同区域
    ↓
找最大连通域 bbox
    ↓
加 margin
    ↓
对所有通道使用同一个 bbox 裁剪
```

这样可以保证最终 `imagesTr` 里面的所有通道：

- size 一致；
- spacing 一致；
- origin 一致；
- direction 一致；
- 每个 voxel 对应同一空间位置。

最终输出类似：

```text
imagesTr/
  CASE001_0000.nii.gz
  CASE001_0001.nii.gz
  CASE001_0002.nii.gz
```

这就是 nnU-Net 多通道输入需要的格式。

---

# 10. 这次样例的结果

这次小样例里，脚本自动选择了切片数最多的 `series003` 作为 fixed。

header 阶段：

- `series004`：z 轴重叠比例太低，`FAIL`
- `series005`：header 通过
- `series006`：header 通过

QC 阶段：

- `series005`：选择 identity，最终 `WARNING`
- `series006`：选择 identity，最终 `WARNING`

它们 warning 的主要原因是质心距离仍然偏大，但 body Dice 和共同区域比例都还可以，所以默认仍然保留进最终多通道。

这个结果其实比第一次尝试合理很多。第一次是 rigid 把结果配坏，然后整个流程没有 moving 可以保留；第二次至少能识别出 rigid 不如 identity，并选择相对更稳的结果。

---

# 11. 目前这个脚本还不完美

当前脚本更像是一个批量化配准和 QC 的实验框架，还不是最终生产级方案。

后面还需要继续改的地方包括：

1. fixed image 的选择应该结合序列描述，而不是只看切片数；
2. body mask 用 `-500 HU` 阈值比较粗糙，后面可以考虑更稳定的身体区域提取；
3. overlay 现在只取中间层，最好多取几个 z 位置；
4. rigid 的优化参数还需要更多病例测试；
5. 对 `WARNING` 的判定阈值还需要根据数据集分布调整；
6. 需要把 QC 图片和 CSV 结果结合起来做一个更方便的人工复查表。

但至少这次已经解决了一个很关键的问题：

> 批量配准不能只问“有没有做 rigid”，而要问“rigid 之后有没有真的变好”。

这也是我这次最大的收获。
