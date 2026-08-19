# 多模态检索平台 — 技术架构文档

## 1. 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    前端 (Modern SPA)                          │
│     React 18 + TypeScript + Vite + TailwindCSS + shadcn/ui    │
│          Zustand + React Query                                │
└──────────────────────────┬──────────────────────────────────┘
                           │ REST API / SSE (流式)
┌──────────────────────────▼──────────────────────────────────┐
│                    API Gateway / 后端                         │
│               FastAPI + JWT + CORS + DI                      │
├───────────┬───────────┬────────────┬───────────┬────────────┤
│ 用户服务   │ 文档服务   │ 检索服务    │ RAG 服务   │ 模型管理    │
│ 认证/权限  │ 上传/解析  │ ES 混合检索 │ 问答/对话  │ 配置/路由   │
│ CRUD      │ 分块/索引  │ 多语言     │ 流式输出   │ 健康检查    │
└───────────┴───┬───┴───┴────────────┴───────────┴────────────┘
                │   │
     ┌──────────┼───┼──────────────────────┐
     │          │   │                      │
     ▼          ▼   ▼                      ▼
┌─────────┐ ┌────────┐ ┌─────────┐ ┌──────────────┐
│PostgreSQL│ │  ES 8  │ │  MinIO  │ │ 模型抽象层    │
│ 元数据   │ │向量+文本│ │ 文件    │ │ (可插拔)      │
└─────────┘ └────────┘ └─────────┘ └──────┬───────┘
                                          │
                    ┌─────────────────────┼───────────────────┐
                    ▼                     ▼                   ▼
              ┌──────────┐        ┌──────────┐        ┌──────────┐
              │  OpenAI  │        │ 通义千问  │        │ DeepSeek │
              │ GPT-4o   │        │ Qwen     │  ...   │ 等       │
              └──────────┘        └──────────┘        └──────────┘
```

## 2. 前端技术栈

| 类别 | 技术 | 说明 |
|------|------|------|
| 框架 | React 18 + TypeScript | 组件化 + 类型安全 |
| 构建 | Vite 5 | 极速 HMR |
| 样式 | TailwindCSS 3 + shadcn/ui | 原子化 CSS + 可定制组件 |
| 状态管理 | Zustand | 轻量级全局状态 |
| 服务端状态 | TanStack Query 5 | 缓存、重试、乐观更新 |
| 路由 | React Router 6 | SPA 路由 |
| 请求 | Axios | HTTP 拦截器、Token 注入 |
| 流式渲染 | Fetch SSE | 问答流式输出 |
| 文件上传 | react-dropzone | 拖拽上传 + 进度 |
| Markdown | react-markdown + remark-gfm | 回答渲染 |
| 图标 | Lucide React | 现代图标库 |

> 界面语言固定中文，不引入 i18n 框架。

## 3. 后端技术栈

| 类别 | 技术 | 说明 |
|------|------|------|
| 框架 | FastAPI 0.110+ | 异步高性能，自动 OpenAPI |
| 语言 | Python 3.13 | |
| 认证 | python-jose + bcrypt | JWT + 密码哈希 |
| 验证 | Pydantic v2 | 请求/响应校验 |
| ORM | SQLAlchemy 2 (async) | 异步 ORM |
| 迁移 | Alembic | 数据库版本管理 |
| 异步任务 | Celery 5 + Redis | 文档处理异步化（--pool=solo） |
| 缓存/队列 | Redis 7 | Celery broker + Token 黑名单 + 缓存 |
| 文档解析 | marker-pdf + pymupdf | 多格式解析 + 矢量图检测 |
| 表格提取 | marker (内置) | PDF 表格结构化抽取 |
| 图像描述 | 多模态 LLM | 生成图像摘要用于索引 |
| 加密 | cryptography (Fernet) | API Key AES 加密存储 |

## 4. 数据存储

| 类别 | 技术 | 说明 |
|------|------|------|
| 关系数据库 | PostgreSQL 16 | 用户、知识库、文档、对话、模型配置 |
| 向量+文本 | Elasticsearch 8.x | 文本块 + 稠密向量 + 元数据统一索引 |
| 对象存储 | MinIO | 原始文件、抽取图片、缩略图 |
| 缓存/队列 | Redis 7 | Celery broker、Token 黑名单、热点缓存 |

## 5. 模型抽象层（可插拔核心）

```
                    ┌──────────────────┐
                    │   ModelRegistry  │
                    │   (模型注册中心)   │
                    └────────┬─────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  LLMProvider  │  │EmbedProvider  │  │VisionProvider │
│   (接口)       │  │  (接口)        │  │  (接口)        │
└───────┬───────┘  └───────┬───────┘  └───────┬───────┘
        │                  │                   │
   ┌────┴────┐       ┌────┴────┐        ┌────┴────┐
   ▼    ▼    ▼       ▼    ▼    ▼        ▼    ▼    ▼
 OpenAI Qwen DS   BGE-M3 OpenAI Cohere  CLIP SigLIP ...
```

| 模型类型 | 接口定义 | 已支持提供商 |
|---------|---------|------------|
| `LLMProvider` | generate(stream), chat(messages) | OpenAI, 通义千问, DeepSeek, Claude, Gemini |
| `TextEmbedProvider` | embed(texts) → vectors | BGE-M3, OpenAI text-embedding, Cohere, E5, mUSE |
| `VisionEmbedProvider` | embed(images) → vectors | CLIP, SigLIP, OpenCLIP |
| `VisionLLMProvider` | describe(image, prompt) → text | GPT-4o, Qwen-VL, Gemini |

### 接口定义

```python
class LLMProvider(ABC):
    @abstractmethod
    async def chat(self, messages: list[dict], **kwargs) -> str: ...

    @abstractmethod
    async def chat_stream(self, messages: list[dict], **kwargs) -> AsyncGenerator[str, None]: ...

class TextEmbedProvider(ABC):
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    @property
    @abstractmethod
    def dim(self) -> int: ...

class VisionEmbedProvider(ABC):
    @abstractmethod
    async def embed(self, images: list[bytes]) -> list[list[float]]: ...

    @property
    @abstractmethod
    def dim(self) -> int: ...

class VisionLLMProvider(ABC):
    @abstractmethod
    async def describe(self, image: bytes, prompt: str) -> str: ...
```

## 6. 文档处理流程

### 6.1 解析策略（两次解析）

```
PDF 输入
  │
  ├─ 第一次：marker fast 模式
  │   ├─ 文本层提取（pdftext）
  │   ├─ 位图提取（Image XObject）
  │   ├─ 布局分析（rf-detr 20M 模型）
  │   └─ 输出：markdown + 图片列表
  │
  ├─ 第二次：矢量图检测（pymupdf）
  │   ├─ get_drawings() 获取 Path+Text+Shape
  │   ├─ 聚类相邻矩形 → 图形区域
  │   ├─ 3x 分辨率裁剪渲染为 PNG
  │   └─ 输出：矢量图列表 + caption
  │
  └─ 退化检测
      ├─ 文本 < 100 字符 → 扫描件 → 整页渲染
      └─ llama-server 不可用 → 整页渲染
```

### 6.2 分块策略（图片独立成块）

```
markdown 文本
  │
  ├─ 文本分块：chunk_by_paragraph()
  │   └─ 按 \n\n 分段，不切散句子
  │
  └─ 图片分块：每个 ![alt](path) 独立成 chunk
      ├─ content = caption（图片后第一行非空文本）
      ├─ chunk_type = "image"
      └─ metadata = {image_info, image_url, image_path}

最终 chunks = text_chunks + image_chunks（重新编号）
```

### 6.3 图片类型支持

| 类型 | 检测方式 | 提取方式 | 格式 |
|------|---------|---------|------|
| 位图（照片/截图） | marker Image XObject | PIL 保存 | PNG |
| 矢量图（架构图/流程图） | pymupdf get_drawings() 聚类 | 裁剪渲染 3x | PNG |
| 扫描件 | 文本<100字符退化 | 整页渲染 2x | PNG |

## 7. Elasticsearch 索引设计

```
Index: kb_{knowledge_base_id}

Document:
{
  "chunk_id": "uuid",
  "doc_id": "uuid",
  "kb_id": "uuid",
  "user_id": "uuid",
  "chunk_index": 0,
  "chunk_type": "text|image|title",
  "content": "段落文本... 或 图片caption...",
  "language": "zh|en|ja|...",
  "dense_vector": [0.01, ...],       // 可配置维度
  "image_url": "minio://bucket/...",
  "image_caption": "图片描述...",
  "images": [{url, path, caption}],  // 图片数组
  "page_number": 3,
  "section_heading": "第三章 ...",
  "document_name": "论文.pdf",
  "document_path": "minio://bucket/...",
  "metadata": {...},
  "created_at": "2026-08-16T..."
}
```

### 检索策略

- `dense_vector` → kNN 语义检索（跨语言对齐）
- `content` → BM25 关键词检索
- `images.caption` → nested 查询（图片 caption 检索）
- 融合：RRF（Reciprocal Rank Fusion）应用层融合
- 过滤：`user_id` + `kb_id` 严格隔离

## 8. 多语言方案

| 层面 | 方案 |
|------|------|
| 多语言嵌入 | BGE-M3（原生支持 100+ 语言，统一语义空间） |
| 跨语言检索 | 多语言查询 → 多语言向量空间 → 检索任意语言内容 |
| 问答语言 | 用户可用任意语言提问，系统自动处理 |
| 文档语言 | 自动检测文档语言，索引时标记 |

## 9. 部署架构

| 类别 | 技术 |
|------|------|
| 容器 | Docker + Docker Compose |
| 反向代理 | Nginx |
| 监控 | Prometheus + Grafana（预留） |
| CI/CD | GitHub Actions（预留） |

## 10. 项目目录结构

```
multimodal/
├── docker-compose.yml
├── .env.example
├── README.md
├── docs/
│   ├── requirements.md
│   ├── architecture.md
│   └── TODO.md
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── auth.py
│   │   │   ├── kb.py
│   │   │   ├── search.py
│   │   │   ├── chat.py
│   │   │   └── model_config.py
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   ├── security.py
│   │   │   ├── dependencies.py
│   │   │   └── celery_app.py
│   │   ├── models/
│   │   ├── schemas/
│   │   ├── services/
│   │   │   ├── doc_parser.py      ← marker + pymupdf 解析
│   │   │   ├── chunker.py         ← 段落分块 + 图片独立成块
│   │   │   ├── indexer.py         ← ES 索引
│   │   │   ├── retriever.py       ← 混合检索 + RRF
│   │   │   ├── rag_pipeline.py
│   │   │   ├── minio_client.py
│   │   │   └── model_manager.py
│   │   ├── models_lib/
│   │   └── main.py
│   ├── alembic/
│   ├── tests/
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── api/
│   │   ├── components/
│   │   ├── pages/
│   │   │   ├── Login.tsx
│   │   │   ├── Register.tsx
│   │   │   ├── Dashboard.tsx      ← 知识库删除功能
│   │   │   ├── KBDetail.tsx
│   │   │   ├── Search.tsx         ← 图片 nested 查询
│   │   │   └── Chat.tsx
│   │   ├── stores/
│   │   ├── hooks/
│   │   ├── types/
│   │   ├── utils/
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   ├── tailwind.config.ts
│   ├── vite.config.ts
│   ├── tsconfig.json
│   └── Dockerfile
└── nginx/
    └── nginx.conf
```

## 11. 已知问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| `daemonic processes are not allowed to have children` | celery prefork pool 创建 daemon 进程，无法 spawn 子进程 | `--pool=solo` 单进程运行 |
| marker 无法提取矢量图 | 矢量图 = Path+Text+Shape，无 Image XObject | pymupdf `get_drawings()` 聚类 + 裁剪渲染 |
| 图片与 caption 分离 | 分块器按 `\n\n` 切分，图片和 caption 被拆到不同 chunk | 图片独立成块，caption 作为 chunk content |
| caption 正则不匹配空 alt | `[^\\]]` 要求至少 1 个字符 | 改为 `[^\]]` |
| ES 序列化失败 | metadata 含 bytes（图片 data） | `pop("data")` 移除 bytes |
| llama-server 找不到 | 环境变量未设置 | `LLAMA_CPP_BINARY` 自动检测 |
