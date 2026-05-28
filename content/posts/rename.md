+++
title = 'Rename'
date = '2026-05-28T20:11:04+08:00'
draft = false
description = ''
tags = ['pydicom','数据预处理','脚本']
categories = ['医学图像处理']
+++

# 如何处理乱码DICOM文件夹

---

# 0. 背景

其实本来没想做这个的，只是发现每次去找series_number的时候，我都要一个一个用ITK打开dicom文件，笔者属实是懒于应付这种重复机械的工作，所以写了个批量重命名的脚本~~虽然是生成的并且实际上花费的时间早就够我一个一个点进去看了~~。

每一个乱码文件夹里面可能是一组DICOM序列，比如 CT 的某个扫描序列、localizer 定位像、增强期图像等。问题就是这些文件夹名称没有任何意义，所以写了个python脚本用来批量读取每个DICOM序列的header信息，并且根据series_number和series_description自动重命名文件夹。

完整代码见 Github:[allocate-dicom-renamer](https://github.com/lnk-k/allocate-dicom-renamer)

---

# 1. 脚本目标

这个脚本主要完成三件事：

1. 批量遍历所有病例文件夹；

2. 读取每个 DICOM 序列的 header 信息；

3. 根据 `SeriesNumber` 和是否为 `localizer` 自动重命名文件夹。

最终效果类似：

```text

乱码文件夹1 -> series001

乱码文件夹2 -> series002

乱码文件夹3 -> localizer003

```

---

# 2. 用到的库

```python

import os

import json

import pydicom

import SimpleITK as sitk

```

其中：

- `os`：遍历文件夹、重命名文件夹；

- `json`：保存重命名日志；

- `pydicom`：读取 DICOM header；

- `SimpleITK`：作为备用方式读取 DICOM 信息。

安装依赖：

```bash

pip install pydicom SimpleITK

```

---

# 3. 主要参数

```python

DATA_DIR = "/你的/data/路径"

DRY_RUN = False

MIN_SERIES_IMAGES = 10

```

## DATA_DIR

表示数据总目录，下面应该是多个病例文件夹。

```text

data/

├── case1/

├── case2/

└── case3/

```

## DRY_RUN

这是一个安全开关。

```python

DRY_RUN = True

```

表示只预览，不真正改名。

```python

DRY_RUN = False

```

表示真正执行重命名。

第一次运行建议先设置为 `True`，确认没问题后再改成 `False`。

## MIN_SERIES_IMAGES

用于辅助判断某个序列是不是 localizer。  

一般 localizer / scout 定位像的 DICOM 数量比较少，所以脚本中设置：

```python

MIN_SERIES_IMAGES = 10

```

如果某个序列的 DICOM 文件数少于 10，就倾向于认为它是 localizer。

---

# 4. 核心逻辑

脚本整体流程是：

```text

遍历 data 目录

    ↓

遍历每个 case

    ↓

查找 case 下的 DICOM 序列文件夹

    ↓

读取 DICOM header

    ↓

提取 SeriesNumber、SeriesDescription、ImageType

    ↓

判断是否为 localizer

    ↓

生成新文件夹名

    ↓

重命名并保存日志

```

---

# 5. 读取 DICOM 信息

脚本优先使用 `pydicom` 读取：

```python

ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)

```

这里：

- `stop_before_pixels=True`：只读 header，不读像素数据，速度更快；

- `force=True`：即使 DICOM 格式不完全标准，也尽量读取。

如果 `pydicom` 读取失败，再尝试使用 `SimpleITK` 读取 metadata。

这样做是为了提高脚本对不同 DICOM 数据的兼容性。

---

# 6. 一个小坑： macOS 隐藏文件

macOS 下有时会生成类似这样的隐藏文件：

```text

._xxx.dcm

```

这种文件并不是真正的 DICOM 文件，读取时容易报错。

所以脚本中加入了过滤：

```python

if name.startswith("._"):

    continue

```

---

# 7. 文件夹命名规则

脚本会读取 DICOM 中的 `SeriesNumber`。

如果是普通序列，就命名为：

```text

series001

series002

series003

```

如果判断为 localizer，就命名为：

```text

localizer001

localizer002

localizer003

```

判断 localizer 的依据主要有两个：

1. DICOM 文件数量少于 `MIN_SERIES_IMAGES`；

2. `SeriesDescription` 或 `ImageType` 中包含关键词。

关键词包括：

```text

localizer

scout

topogram

surview

locator

```

---

# 8. 防止重名覆盖

如果目标文件夹已经存在，脚本不会直接覆盖，而是自动加后缀。

例如：

```text

series001

series001_01

series001_02

```

这样可以避免误覆盖已有数据。

---

# 9. 日志保存

重命名完成后，脚本会保存 JSON 日志。

每个 case 下会生成：

```text

rename_series_log.json

```

总目录下会生成：

```text

rename_all_cases_log.json

```

日志中会记录：

```text

原文件夹名

新文件夹名

DICOM 数量

SeriesNumber

SeriesDescription

ImageType

SeriesInstanceUID

```

这样即使文件夹被改名，也可以追溯原始信息。

---

# 10. 使用方法

## 第一步：安装依赖

```bash

pip install pydicom SimpleITK

```

## 第二步：修改路径

```python

DATA_DIR = "/你的/data/路径"

```

## 第三步：先预览

```python

DRY_RUN = True

```

运行：

```bash

python rename.py

```

## 第四步：确认后真正执行

```python

DRY_RUN = False

```

再次运行：

```bash

python rename.py

```

---

# 11. 总结

这个脚本解决的是 DICOM 数据整理中的文件夹命名混乱问题。

它的核心思路是：

```text

不看原始文件夹名，而是读取 DICOM header 信息来重命名。

```

整理后的数据结构更加清晰，方便后续进行 DICOM 转 NIfTI、图像配准以及 nnU-Net 相关实验。