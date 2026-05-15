# 个人网站

这是一个可直接部署到 GitHub Pages 的静态个人网站模板，包含首页、关于、项目、经历和联系方式。

## 修改内容

在以下文件中替换占位信息：

- `index.html`：姓名、简介、项目、邮箱、GitHub 用户名
- `styles.css`：颜色、排版、间距
- `script.js`：页面交互

当前模板已使用 GitHub 用户名 `lnk-k`。如果要换成其他账号，请替换 `index.html` 里的 GitHub 链接。

```html
https://github.com/octocat.png
https://github.com/octocat
```

## 本地预览

直接用浏览器打开 `index.html`，或在目录中运行：

```bash
python3 -m http.server 8000
```

然后访问 `http://localhost:8000`。

## 发布到 GitHub Pages

1. 在 GitHub 新建一个仓库，例如 `lnk-k.github.io`。
2. 把本项目推送到该仓库。
3. 打开仓库的 `Settings` → `Pages`。
4. 在 `Build and deployment` 中选择 `GitHub Actions`。
5. 推送到 `main` 分支后，GitHub 会自动发布网站。

如果仓库名是 `lnk-k.github.io`，网站地址通常是：

```text
https://lnk-k.github.io
```
