# 多模态检索平台

多模态智能检索与 RAG 问答平台，支持文本、图像、表格等多种模态的统一理解与检索。

## 功能特性

- 🔐 用户认证与知识库隔离
- 📄 多格式文档上传与解析（PDF / DOCX / PPTX / MD / 图片等）
- 🌐 多语言检索（中文查询检索英文等内容）
- 🔍 多模态混合检索（语义 + 关键词 + 图像）
- 💬 多模态 RAG 问答（流式输出、溯源引用）
- 🧩 可插拔模型层（LLM / Embedding / 视觉模型灵活切换）
- 🐳 Docker 一键部署

## 文档

- [需求文档](docs/requirements.md)
- [技术架构](docs/architecture.md)

## 技术栈

| 层级 | 技术 |
|------|------|
| 前端 | React 18 + TypeScript + Vite + TailwindCSS + shadcn/ui |
| 后端 | FastAPI + Python 3.13 |
| 数据库 | PostgreSQL 16 |
| 向量+文本 | Elasticsearch 8.x |
| 对象存储 | MinIO |
| 缓存/队列 | Redis 7 |
| 部署 | Docker + Docker Compose + Nginx |

## 快速开始

```bash
# 1. 克隆项目
git clone <repo-url> && cd multimodal

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 API Keys

# 3. 启动服务
docker-compose up -d

# 4. 访问
# 前端: http://localhost
# API 文档: http://localhost/api/docs
```

## 模型配置

平台支持以下模型提供商，可在管理后台配置：

| 类型 | 支持提供商 |
|------|-----------|
| LLM | OpenAI、通义千问、DeepSeek、Claude、Gemini |
| 文本嵌入 | BGE-M3、OpenAI、Cohere、E5、mUSE |
| 视觉嵌入 | CLIP、SigLIP、OpenCLIP |
| 多模态 LLM | GPT-4o、Qwen-VL、Gemini |

## License

MIT
