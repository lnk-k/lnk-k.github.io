+++
title = 'C++ 数据结构实践：七种容器实现与 GoogleTest 测试'
date = '2026-09-22T10:59:30+08:00'
draft = false
description = '用 C++17 实现动态数组、链表、栈、队列、二叉搜索树、B 树和 B+ 树，附完整 CMake 工程、53 个 GoogleTest 用例及源码压缩包。'
tags = ['C++', '数据结构', 'CMake', 'GoogleTest', 'B树', 'B+树']
categories = ['数据结构']
+++

这份项目参考 Java 集合的基本操作，用 C++17 实现七种常见容器，并提供 CMake 工程、演示程序和 GoogleTest 测试。

## 完整项目下载

**[下载 C++ 容器项目（ZIP，约 876 KiB）](/downloads/cpp-containers.zip)**

压缩包包含：

- 七种容器的完整 C++ 源码；
- `CMakeLists.txt` 构建文件；
- GoogleTest 测试代码及官方 v1.15.2 源码包；
- 演示程序、中文 README 和测试报告。

完整项目可离线构建测试，前提是本机已经安装 CMake、C++ 编译器和相应构建工具。

## 实现了哪些容器

| 容器 | 基本操作与实现方式 |
| --- | --- |
| 动态数组 `DynamicArray<T>` | 连续存储、自动扩容、按下标访问、插入和删除 |
| 双向链表 `LinkedList<T>` | 头尾插入删除、按位置访问、删除匹配元素 |
| 栈 `Stack<T>` | 基于动态数组，支持 `push`、`pop`、`top` |
| 队列 `Queue<T>` | 基于链表，支持 `enqueue`、`dequeue`、`front` |
| 二叉搜索树 `BinarySearchTree<T>` | 查找、插入、删除、中序遍历、最小和最大键 |
| B 树 `BTree<T, t>` | 多路平衡搜索，包含结点分裂、借位、合并和根收缩 |
| B+ 树 `BPlusTree<T, t>` | 数据保存在叶子，维护叶子链表，额外支持闭区间范围查询 |

数组、链表、栈和队列允许重复元素；三种树采用不重复键的集合语义。树的 `insert` 和 `erase` 返回操作是否实际改变了集合。

Java 的 `TreeSet` 和 `TreeMap` 基于红黑树，与这里实现的普通二叉搜索树、B 树和 B+ 树不同。项目参考的是基本容器操作，而非复刻 Java 标准库的全部接口。

## 编译和运行

环境要求为支持 C++17 的 GCC、Clang 或 MSVC，以及 CMake 3.16 及以上版本。在解压后的 `cpp-containers` 目录执行：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --config Debug --parallel
ctest --test-dir build -C Debug --output-on-failure
```

上面的 `ctest --test-dir` 写法要求 CMake 3.20 及以上版本；使用 3.16～3.19 时，进入 `build` 目录后执行 `ctest -C Debug --output-on-failure`。

macOS / Linux 运行演示：

```bash
./build/containers_demo
```

Windows 使用 Visual Studio 生成器时：

```powershell
.\build\Debug\containers_demo.exe
```

## B+ 树使用示例

```cpp
#include "containers/containers.hpp"
#include <iostream>

int main() {
    containers::BPlusTree<int, 2> tree;
    for (int x : {40, 10, 30, 20, 50}) {
        tree.insert(x);
    }
    tree.erase(30);

    for (int x : tree.range_query(15, 45)) {
        std::cout << x << ' ';
    }
    // 输出：20 40
}
```

模板参数 `t` 表示最小度数，不能直接理解为最大孩子数。具体结点容量规则和算法说明见压缩包中的 README。

## 测试结果

项目已在 macOS / AppleClang 21、CMake 3.31.6 环境下实际编译运行：**53 个 GoogleTest 用例全部通过**，启用的 AddressSanitizer 和 UndefinedBehaviorSanitizer 未报告错误。

测试覆盖空容器、越界、扩容、深拷贝、移动、对象生命周期、树的插入删除和 B+ 树范围查询。三种树使用标准库 `std::set` 作为参考，每次随机操作后检查结构、大小和完整遍历结果；B 树和 B+ 树分别测试了最小度数为 2、3、8 的配置。

```text
100% tests passed, 0 tests failed out of 53
```

这是面向基本数据结构操作的教学实现。普通 BST 不保证平衡；B+ 树为了便于理解，在更新时重建分隔键，其更新复杂度的保守上界为 O(t h²)，其中 h 为树高。项目没有实现并发访问、磁盘持久化或完整 STL 接口。

## 文件校验

ZIP 文件的 SHA-256：

```text
3bd83b0764492d4698c1225b331c1595d77d6119c18a76befb650d75c0a1840d
```

GoogleTest 的原始许可证保留在项目的 `third_party/LICENSE.googletest` 中。
