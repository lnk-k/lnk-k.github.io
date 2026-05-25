+++
title = 'Registration'
date = '2026-05-25T15:34:27+08:00'
draft = false
description = ''
tags = ['期相配准']
categories = ['医学图像识别']
+++

# 医学图像配准学习笔记

这周继续研究期相配准。。

## 1.图像配准定义

根据笔者粗浅的理解，配准就是通过空间变换将 moving image 变换到fixed image上，使两个图像上中的公共点一一对应。

目前我的任务就是将同一个病例中不同 DICOM 序列的图像统一到同一个空间坐标系中，使它们具有一致的：

- spacing：体素间距

- origin：原点坐标

- direction：方向矩阵

- size：图像尺寸

最终目标是让不同序列可以作为 nnU-Net 的多通道输入，因为这些通道必须在空间上对应，否则模型学到的多序列信息就会错位。

---

## 2.图像配准应用场景

这一部分就不必多言，因为笔者本身做的项目也就只和图像识别有关。

---

## 3. 当前任务的配准类型定位

虽然图像配准有很多分类方式，例如按模态、对象、变换类型等分类，但对当前任务来说，不需要展开全部分类。

当前任务已经比较明确：

- 应用领域：医学图像配准

- 图像类型：CT 图像

- 配准对象：同一病例内部不同期相

- 配准类型：多期相 CT 配准

- 图像维度：3D 到 3D 配准

- 目标用途：统一不同期相图像空间，使其适配 nnU-Net 多通道输入

因此，当前任务可以概括为：

> 同一病例多期相 CT 图像之间的 3D 医学图像配准。

在这个任务中，重点不是讨论各种配准分类，而是解决以下问题：

1. 如何选择 fixed image 和 moving image；

2. 如何让不同期相的图像在空间上对齐；

3. 如何保证配准后同编号切片对应同一解剖位置；

4. 如何处理不同期相扫描范围不一致的问题；

5. 如何将处理结果转换为 nnU-Net 可用的多通道输入格式。

从方法选择上看，当前任务主要关注：

- 重采样 Resampling

- 刚性配准 Rigid Registration

- 仿射配准 Affine Registration

- 必要时考虑非刚性配准 Deformable Registration

其中，当前阶段优先考虑：

> 刚性配准 + 重采样 + 共同有效区域裁剪

原因是刚性配准可以校正不同期相之间的整体平移和旋转，同时不会改变器官和病灶的真实形态。对于医学图像分割前处理来说，这种方法相对稳定、安全，也更适合作为批量化处理流程的基础。

如果刚性配准后仍然存在整体尺度差异，可以进一步尝试仿射配准；如果存在明显局部形变，再谨慎考虑非刚性配准。

---

## 4.图像配准过程

### 4.1 配准中的核心概念

#### 4.1.1 Fixed Image

`fixed image` 是基准图像，提供统一的空间标准，包括：

- spacing

- origin

- direction

- size

- 切片编号对应的空间位置

在当前任务中，一般选择扫描范围较完整、图像质量较好的序列作为 fixed image。

---

#### 4.1.2 Moving Image

`moving image` 是待配准图像，也就是需要对齐到 fixed image 的其他期相或序列。

配准的目标就是让 moving image 中的解剖结构尽量和 fixed image 对应。

---

#### 4.1.3 Transform

`transform` 是空间变换，用来描述 moving image 应该如何移动到 fixed image 上。

当前任务中主要关注：

- 刚性变换：平移 + 旋转

- 仿射变换：平移 + 旋转 + 缩放 + 剪切

- 非刚性变换：局部形变

当前阶段优先考虑刚性配准，必要时再尝试仿射配准。非刚性配准可能改变病灶形态，因此需要谨慎使用。

---

#### 4.1.4 Metric

`metric` 是相似性度量指标，用来判断 fixed image 和变换后的 moving image 是否对齐。

常见指标包括：

- MSE：均方误差

- NCC：归一化互相关

- MI：互信息

对于多期相 CT，不同期相的灰度分布可能不同，因此互信息 `Mutual Information` 通常比直接比较灰度差异更稳妥。

---

#### 4.1.5 Optimizer

`optimizer` 是优化器，用来不断调整 transform 的参数，使 metric 结果变得更好。

简单理解就是：

算法不断尝试不同的平移、旋转或缩放参数，直到找到较好的对齐方式。

---

#### 4.1.6 Interpolator

`interpolator` 是插值器。

当 moving image 经过空间变换后，新的体素位置不一定刚好落在原体素点上，因此需要插值计算新的灰度值。

当前任务中：

- 原始 CT/MRI 图像：使用线性插值

- 标签 mask：使用最近邻插值

标签不能用线性插值，否则类别值可能变成小数。

---

### 4.2 配准的一般流程

图像配准通常包括：

1. 特征检测

2. 特征匹配

3. 变换模型估计

4. 图像重采样和变换

结合当前任务，可以简化理解为：

| 步骤 | 当前任务中的含义 |

|---|---|

| 特征检测 | 不手动找特征点，主要依靠图像整体结构和灰度统计 |

| 特征匹配 | 判断不同期相中的相同解剖结构是否对应 |

| 变换模型估计 | 估计 moving image 到 fixed image 的空间变换 |

| 重采样和变换 | 将 moving image 放到 fixed image 的空间网格中 |

当前最关键的是：

- 估计空间变换

- 重采样到 fixed image 空间

- 检查同编号切片是否对应同一部位

---

### 4.3 特征检测与特征匹配

在通用图像配准中，特征可以是角点、边缘、线段或控制点。

但在当前医学 CT 任务中，不重点依赖人工特征点。原因是：

1. 医学图像中稳定明显的角点不一定多；

2. 不同期相 CT 中血管和病灶强化不同，局部特征可能不稳定；

3. 当前目标是批量处理，不适合人工点选特征。

因此，当前更适合使用基于图像整体灰度和统计关系的自动配准方法。

特征匹配在当前任务中可以理解为：

判断两个期相中相同身体位置的结构是否对齐，例如肺、肝脏、肾脏、骨骼等是否对应。

---

### 4.4 相似性度量方法

配准过程中需要判断 fixed image 和 moving image 是否对齐，这就需要相似性度量。

对于灰度分布接近的图像，可以使用：

- MSE

- NCC

但对于多期相增强 CT，不同期相的灰度表现不同，例如动脉期血管更亮、门静脉期肝实质强化更明显，因此更适合使用：

- Mutual Information，互信息

互信息不要求两个图像灰度值完全相同，而是衡量两幅图像之间的统计相关性，因此更适合多期相 CT 配准。

---

### 4.5 图像重采样和变换

当 transform 估计完成后，需要将 moving image 重采样到 fixed image 的空间网格中。

重采样后的图像应该和 fixed image 具有一致的：

- spacing

- origin

- direction

- size

但需要注意：

> 重采样成功不等于图像内容一定配准成功。

所以除了检查 header 是否一致，还需要通过 overlay 或 checkerboard 检查图像内容是否真正对齐。

---

### 4.6 共同覆盖区域裁剪

如果两个期相扫描范围不同，例如：

- fixed image 有 140 张，从锁骨扫到腹部

- moving image 有 90 张，从肺部扫到腹部

那么 moving image 重采样到 fixed image 后，可能会出现一部分空白区域。

这种情况下，需要在配准和重采样后，裁剪两个期相的共同有效区域。

正确流程是：

1. 先将 moving image 配准并重采样到 fixed image 空间

2. 再寻找共同覆盖区域

3. 对所有通道使用同一个裁剪框

4. 输出给 nnU-Net

不能让每个期相单独裁剪，否则会破坏不同通道之间的空间对应关系。

---

### 4.7 当前任务推荐流程

当前任务中，比较合理的流程是：

1. 读取同一病例下的多个 DICOM 序列

2. 按 `SeriesInstanceUID` 区分不同序列

3. 排除 localizer、scout、切片数过少或不同部位的序列

4. 选择一个序列作为 fixed image

5. 其他序列作为 moving image

6. 使用刚性配准或必要时仿射配准

7. 使用互信息作为多期相 CT 的相似性指标

8. 将 moving image 重采样到 fixed image 空间

9. 裁剪共同覆盖区域

10. 检查同编号切片是否对应同一部位

11. 保存为 nnU-Net 多通道 NIfTI 格式

---

### 4.8 小结

图像配准过程可以概括为：

> fixed image 提供空间标准，moving image 通过 transform 对齐到 fixed image；metric 判断是否对齐，optimizer 寻找最佳参数，interpolator 完成重采样过程。

对于当前多期相 CT 任务，重点是：

- 优先使用刚性配准

- 必要时使用仿射配准

- 相似性指标优先使用互信息

- 重采样后仍需检查图像内容是否对齐

- 如果扫描范围不同，需要裁剪共同覆盖区域

- 最终目标是让不同期相同编号切片对应同一身体位置，并适配 nnU-Net 多通道输入

## 5.总结

虽然扯了这么多，不过实际上只是提出了方案也没有落到实处，后面会再详细去学习具体的方法。。。。

## 参考文献

[1] CSDN 博客. 图像配准相关学习笔记/教程[EB/OL].  
https://blog.csdn.net/qq_31347869/article/details/119394395  
访问日期：2026-05-25.

[2] Zitová B, Flusser J. Image registration methods: a survey[J]. Image and Vision Computing, 2003, 21(11): 977-1000.

[3] SimpleITK Documentation. Image Registration Method[EB/OL].  
https://simpleitk.readthedocs.io/en/master/link_ImageRegistrationMethod1_docs.html  
访问日期：2026-05-25.

[4] SimpleITK Documentation. ImageRegistrationMethod Class Reference[EB/OL].  
https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1ImageRegistrationMethod.html  
访问日期：2026-05-25.

[5] Isensee F, Jaeger P F, Kohl S A A, Petersen J, Maier-Hein K H. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation[J]. Nature Methods, 2021, 18: 203-211.