+++
title = '第一篇博客'
date = 2026-05-16T00:00:00+08:00
draft = false
description = '这个博客已经用 Hugo 和 GitHub Pages 搭建完成。'
tags = ['Hugo', 'GitHub Pages']
categories = ['博客搭建']
+++

这是博客的第一篇文章。

现在这个仓库已经变成 Hugo 项目：以后你只需要在 `content/posts/` 目录里新增 Markdown 文件，推送到 GitHub 后，GitHub Actions 会自动构建并发布到 GitHub Pages。

本地新建文章可以运行：

```bash
hugo new content/posts/my-new-post.md
```

本地预览可以运行：

```bash
hugo server -D
```

然后打开 `http://localhost:1313/`。
