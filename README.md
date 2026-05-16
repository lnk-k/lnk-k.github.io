# lnk-k 的 Hugo 博客

这是一个使用 Hugo 构建，并通过 GitHub Actions 自动部署到 GitHub Pages 的个人博客。

线上地址：

```text
https://lnk-k.github.io/
```

## 本地预览

安装 Hugo 后运行：

```bash
hugo server -D
```

然后访问：

```text
http://localhost:1313/
```

## 新建文章

```bash
hugo new content/posts/my-new-post.md
```

编辑文章 front matter，将 `draft = true` 改为 `draft = false` 后即可发布。

## 发布到 GitHub Pages

推送到 `main` 分支后，`.github/workflows/hugo.yaml` 会自动构建并发布博客。

## 目录结构

```text
content/posts/       博客文章
content/about.md     关于页面
layouts/             Hugo 模板
assets/css/main.css  站点样式
hugo.toml            Hugo 配置
```
