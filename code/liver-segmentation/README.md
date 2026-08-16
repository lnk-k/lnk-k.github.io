# 肝脏 CT 识别与分割实验

这是我从肝脏粗 ROI 定位走向像素级分割的实验代码。项目先尝试了监督式对比学习 patch 分类器，再切换到轻量 2D U-Net；当前主线是 U-Net。

> 这是个人学习与研究代码，不是官方 nnU-Net 实现，也不能用于临床诊断。

## 当前进度

截至 2026-08-14，轻量 2D U-Net 已完成一次完整 baseline：

- 数据集：SLIVER07，共 20 例带肝脏标签的 CT。
- 划分：16 例训练、4 例整卷验证，按病例隔离。
- 训练输入：3,134 张轴向切片，统一为 `256 × 256`。
- 模型：单通道 2D U-Net，base channels 为 16，共 1,942,289 个参数。
- 训练：30 epochs，AdamW，Soft F2 loss；每轮在 4 个完整 CT 上验证。
- 当前最佳：epoch 30、阈值 0.30。

以下为 4 个验证病例的宏平均结果，均来自未经连通域后处理的 raw mask：

| 方法 | Recall | Precision | Dice |
| --- | ---: | ---: | ---: |
| 旧 SupCon 滑窗方案 | 0.9504 | 0.7266 | 0.8220 |
| 当前 2D U-Net | 0.9837 | 0.9449 | 0.9639 |

U-Net 的整体验证微平均为 Recall 0.9830、Precision 0.9463、Dice 0.9643、F2 0.9755。逐病例结果见 [`results/unet_validation_epoch_030.csv`](results/unet_validation_epoch_030.csv)。

这些数字只代表当前 4 例内部验证集，不是独立测试集结果，也不代表临床性能。

## 为什么从 patch 分类换到 U-Net

旧方案先判断一个 patch “像不像肝脏”，再把滑窗概率叠回整卷 CT。它适合做粗 ROI，但很难直接得到干净边界：

- patch 分类的目标和逐像素分割目标并不一致；
- 概率回填会让边缘天然偏厚、偏模糊；
- 二维局部 patch 缺少整器官上下文；
- 阈值和连通域后处理对最终结果影响很大；
- 高召回通常伴随较多误检。

U-Net 直接学习 image-to-mask，让训练目标和最终任务对齐。当前 baseline 在不做后处理的情况下已经明显降低误检，并保住了较高召回率。

## 目录结构

```text
.
├── scripts/
│   ├── liver_unet.py                 # 轻量 2D U-Net、Soft F2 与指标
│   ├── liver_unet_data.py            # CT 缓存、病例划分、切片 Dataset
│   ├── prepare_liver_unet_data.py    # NIfTI -> uint8/mask NumPy 缓存
│   ├── train_liver_unet.py           # 训练与整卷验证
│   ├── infer_liver_unet.py           # 整卷推理，输出对齐的 NIfTI
│   ├── train_liver_supcon.py         # 旧 SupCon patch baseline
│   ├── infer_liver_heatmap.py        # 旧滑窗概率热图
│   └── ...                           # ROI、切片和 patch 准备工具
├── results/
│   └── unet_validation_epoch_030.csv
├── requirements.txt
└── requirements-train.txt
```

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-train.txt
```

主要依赖：

- NumPy
- SciPy
- SimpleITK
- Pillow
- PyTorch

## 数据目录

代码默认使用下面的结构：

```text
datasets/SLIVER07/processed_for_contrastive/
├── images/
│   ├── sliver07_001_0000.nii.gz
│   └── ...
├── liver_masks/
│   ├── sliver07_001_liver_mask.nii.gz
│   └── ...
└── unet_slice_cache/
```

原始医学影像、标签、训练缓存和模型权重不会提交到 GitHub。请自行确认数据集许可，并确保任何私有病例已经去标识化。

## 1. 准备 U-Net 缓存

CT 先按 `[-100, 250]` HU window 裁剪并映射到 uint8；mask 转成二值数组。图像使用双线性缩放，标签使用最近邻缩放。

```bash
python scripts/prepare_liver_unet_data.py \
  --images-dir datasets/SLIVER07/processed_for_contrastive/images \
  --masks-dir datasets/SLIVER07/processed_for_contrastive/liver_masks \
  --cache-dir datasets/SLIVER07/processed_for_contrastive/unet_slice_cache
```

## 2. 先做小规模 smoke test

正式训练前可以先用少量切片确认数据、损失和设备都能正常工作：

```bash
python scripts/train_liver_unet.py \
  --cache-dir datasets/SLIVER07/processed_for_contrastive/unet_slice_cache \
  --output-dir models/liver_unet_pilot \
  --epochs 3 \
  --max-train-slices 512 \
  --batch-size 4 \
  --val-cases sliver07_001 \
  --device auto
```

## 3. 完整训练

本次 baseline 的训练设置如下：

```bash
python scripts/train_liver_unet.py \
  --cache-dir datasets/SLIVER07/processed_for_contrastive/unet_slice_cache \
  --output-dir models/liver_unet_baseline_full \
  --epochs 30 \
  --batch-size 32 \
  --val-batch-size 64 \
  --image-size 256 \
  --base-channels 16 \
  --learning-rate 0.001 \
  --weight-decay 0.00001 \
  --threshold 0.30 \
  --checkpoint-every 5 \
  --num-workers 4 \
  --device cuda
```

如果没有 CUDA，`--device auto` 会依次尝试 CUDA、Apple MPS 和 CPU。根据显存调整 batch size 即可。

训练会输出：

- `best.pt`、`last.pt` 和定期 checkpoint；
- `train_log.csv`；
- `validation_cases.csv`；
- `config.json`。

最佳模型按 4 个完整验证体数据的 Hard F2 选择，而不是只看切片级 loss。

## 4. 整卷推理

```bash
python scripts/infer_liver_unet.py \
  --checkpoint models/liver_unet_baseline_full/best.pt \
  --input /path/to/case_0000.nii.gz \
  --output-dir outputs/case_0000 \
  --threshold 0.30 \
  --device auto
```

输出：

- `liver_probability.nii.gz`：肝脏概率；
- `liver_mask_raw.nii.gz`：raw 二值 mask；
- `liver_contour_raw.nii.gz`：三维轮廓；
- `inference_summary.json`：checkpoint、阈值和连通域统计。

输出会复制输入 NIfTI 的 spacing、origin 和 direction，以便继续在 3D Slicer 或 ITK-SNAP 中检查。

## 旧的 SupCon 粗 ROI 路线

旧路线仍保留在仓库中，主要用于复现实验对照：

```text
完整 CT + liver mask
        ↓
采样 liver / non-liver patch
        ↓
监督式对比学习 + 二分类
        ↓
整卷滑窗推理
        ↓
概率热力图 + 阈值 + 连通域后处理
```

它的价值是快速验证“肝脏区域能不能被定位”，但当前不会再把它当作最终精细分割方案。

## 下一步

- 增加独立测试集与交叉验证，避免只看 4 个验证病例；
- 加入 Dice + BCE/Focal 等损失对照；
- 做 spacing-aware 重采样和更完整的数据增强；
- 比较 2.5D、3D U-Net 与官方 nnU-Net baseline；
- 增加 HD95、Surface Dice 和逐病例失败分析；
- 最终把肝脏分割作为前置 ROI，继续做肝内病灶识别。
