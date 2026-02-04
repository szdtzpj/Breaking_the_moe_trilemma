# 将项目上传到GitHub的完整指南

## 📋 前提条件

1. **安装Git**
   - 下载地址：https://git-scm.com/download/win
   - 安装时选择"Git from the command line and also from 3rd-party software"
   - 安装完成后重启PowerShell

2. **创建GitHub账号**
   - 访问：https://github.com
   - 如果已有账号，请登录

## 🚀 上传步骤

### 步骤1：安装Git（如果尚未安装）

1. 访问 https://git-scm.com/download/win
2. 下载并安装Git for Windows
3. 安装完成后，重启PowerShell或命令提示符

### 步骤2：配置Git（首次使用）

打开PowerShell，运行以下命令：

```powershell
# 设置你的用户名（替换为你的GitHub用户名）
git config --global user.name "你的用户名"

# 设置你的邮箱（替换为你的GitHub邮箱）
git config --global user.email "your.email@example.com"
```

### 步骤3：在GitHub上创建新仓库

1. 登录GitHub：https://github.com
2. 点击右上角的 "+" 按钮，选择 "New repository"
3. 填写仓库信息：
   - **Repository name**: `Breaking_the_moe_trilemma` 或你喜欢的名字
   - **Description**: "DEGUC: Dynamic Expert Grouping with Unified Compression - Breaking the MoE Trilemma"
   - **Public/Private**: 选择 Public（开源）
   - **不要**勾选 "Initialize this repository with a README"（我们已经有了）
4. 点击 "Create repository"

### 步骤4：初始化本地Git仓库并上传

在PowerShell中，切换到项目目录并运行以下命令：

```powershell
# 切换到项目目录
cd C:\Users\huawei\PycharmProjects\Breaking_the_moe_trilemma

# 初始化Git仓库
git init

# 添加所有文件到暂存区
git add .

# 创建第一次提交
git commit -m "Initial commit: DEGUC implementation"

# 添加远程仓库（替换YOUR_USERNAME为你的GitHub用户名）
git remote add origin https://github.com/YOUR_USERNAME/Breaking_the_moe_trilemma.git

# 推送到GitHub（首次推送）
git push -u origin master
```

**注意**：如果推送时要求输入用户名和密码：
- 用户名：你的GitHub用户名
- 密码：需要使用Personal Access Token（不是GitHub密码）

### 步骤5：创建Personal Access Token（如果需要）

如果推送时需要认证：

1. 访问：https://github.com/settings/tokens
2. 点击 "Generate new token" → "Generate new token (classic)"
3. 设置：
   - Note: "Breaking_the_moe_trilemma"
   - Expiration: 选择有效期
   - 勾选 "repo" 权限
4. 点击 "Generate token"
5. **复制生成的token**（只显示一次！）
6. 在推送时，使用token作为密码

### 步骤6：验证上传成功

1. 访问你的GitHub仓库页面：`https://github.com/YOUR_USERNAME/Breaking_the_moe_trilemma`
2. 确认所有文件都已上传
3. 检查README.md是否正确显示

## 🔄 后续更新代码

当你修改代码后，使用以下命令更新GitHub仓库：

```powershell
# 查看修改的文件
git status

# 添加修改的文件
git add .

# 提交修改
git commit -m "描述你的修改内容"

# 推送到GitHub
git push
```

## 📝 常用Git命令

```powershell
# 查看仓库状态
git status

# 查看提交历史
git log

# 查看远程仓库
git remote -v

# 拉取最新代码
git pull

# 创建新分支
git checkout -b feature-branch-name

# 切换分支
git checkout branch-name

# 合并分支
git merge branch-name
```

## ⚠️ 注意事项

1. **大文件处理**：
   - 如果有大于100MB的文件，需要使用Git LFS
   - 安装：`git lfs install`
   - 追踪大文件：`git lfs track "*.pth"`

2. **敏感信息**：
   - 确保不要上传API密钥、密码等敏感信息
   - 检查 `.gitignore` 文件是否正确配置

3. **数据文件**：
   - 如果数据文件很大，考虑不上传到GitHub
   - 在 `.gitignore` 中添加数据目录

4. **模型检查点**：
   - 训练好的模型文件通常很大
   - 已在 `.gitignore` 中排除
   - 可以使用HuggingFace Hub或其他平台分享模型

## 🎉 完成！

完成以上步骤后，你的项目就成功开源到GitHub了！

你可以：
- 分享仓库链接给其他人
- 在README中添加更多文档
- 接受其他开发者的贡献
- 使用GitHub Issues追踪问题
- 使用GitHub Actions设置CI/CD

## 🆘 遇到问题？

常见问题解决：

1. **"git不是内部或外部命令"**
   - 需要先安装Git
   - 安装后重启PowerShell

2. **推送被拒绝（rejected）**
   - 运行：`git pull origin master --rebase`
   - 然后再推送：`git push`

3. **认证失败**
   - 确保使用Personal Access Token而不是密码
   - 检查token权限是否正确

4. **文件太大无法推送**
   - 使用Git LFS处理大文件
   - 或将大文件添加到 `.gitignore`

---

**祝你开源顺利！** 🚀

