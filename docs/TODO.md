# TODO List

## 📄 文档 (Docs)
- [x] README 更新 — 补充本地开发启动流程（conda 环境 + docker 基础设施 + uvicorn + celery）
- [x] 架构文档更新 — 补充文档解析流程、分块策略、矢量图检测
- [ ] API 文档 — 补充搜索接口 `/api/search` 的请求/响应格式说明
- [ ] 部署文档 — 补充 `--profile docker` 部署方式、elasticsearch-py 版本约束、markitdown[all] 依赖说明
- [ ] 环境变量文档 — 整理 `.env.example` 中所有配置项的说明（含新增的 `MINIO_CONSOLE_PORT`、`VITE_API_BASE_URL`）
- [ ] 故障排查文档 — 记录已知问题（passlib/bc 不兼容、ES 客户端版本、RRF 许可证限制）

---

## 🔧 开发 (Development)

### 用户管理
- [x] 登录功能 — 前端登录页面与后端 `/api/auth/login` 对接
- [ ] 密码重置 — 支持通过邮箱验证重置密码
- [ ] 用户注册优化 — 邮箱验证、密码强度校验
- [ ] 用户资料 — 查看/编辑个人信息（full_name、avatar_url）
- [ ] Token 刷新 — 前端自动刷新 access_token 逻辑

### 知识库管理
- [x] 删除知识库功能 — 前端 hover 显示删除按钮 + 确认弹窗 + 后端级联清理（ES索引/MinIO文件/DB记录）

### 管理员功能
- [ ] 管理员角色 — `is_admin` 字段权限校验中间件
- [ ] 用户管理面板 — 查看/禁用/删除用户
- [ ] 知识库管理 — 管理员可查看/管理所有用户的知识库
- [ ] 系统监控 — 文档处理队列状态、Celery worker 状态
- [ ] 模型配置管理 — 管理员可全局配置默认嵌入模型、LLM 模型

### 检索功能细化
- [ ] 搜索结果高亮 — 前端展示 highlight 片段
- [ ] 检索模式切换 — 前端支持 hybrid / semantic / keyword 三种模式选择
- [ ] 多知识库联合检索 — 支持同时选择多个知识库搜索
- [ ] 搜索结果分页 — top_k 较大时分页展示
- [ ] 检索历史 — 保存用户搜索记录
- [ ] 检索性能优化 — 嵌入模型缓存、批量检索

### 入库效率优化
- [ ] 大文件分片上传 — 避免单次读取超大文件内存溢出
- [x] 异步处理状态推送 — WebSocket 实时通知文档处理进度（已部分实现）
- [ ] 批量上传 — 支持同时上传多个文件
- [ ] 重复文档检测 — 基于文件哈希去重
- [ ] 嵌入模型预热 — Celery worker 启动时预加载模型到内存，避免每次请求重新加载（当前每次 ~25s）
- [ ] 索引批量优化 — bulk 索引批次调优

---

## 🧪 测试 (Testing)

### 多模态检索功能测试
- [ ] 文本检索测试 — 验证 hybrid/semantic/keyword 三种模式返回结果准确性
- [x] 图像检索测试 — 上传含图片的 PDF，验证图片提取和图像-文本联合检索
- [ ] 跨语言检索测试 — 中英文混合文档的检索效果
- [ ] 检索相关性评估 — 构建测试集评估召回率和准确率
- [ ] 性能测试 — 大量 chunks 下的检索响应时间
- [ ] 并发测试 — 多用户同时检索的稳定性

---

## ✅ 已完成（2026-08-19 更新）

### 文档解析与分块
- [x] marker fast 模式 + `--disable_ocr`（纯 CPU，无 OCR，除非检测到扫描件）
- [x] 矢量图检测与提取 — pymupdf `get_drawings()` 聚类 + 裁剪渲染为 PNG
- [x] 图片独立成块 — 文本按段落分块，每个图片+caption 独立成 chunk
- [x] caption 关联修复 — 正则 `[^\\]]` → `[^\]]`，正确匹配空 alt
- [x] 序列化修复 — `metadata["image_info"].pop("data")` 移除 bytes
- [x] daemon 进程修复 — celery `--pool=solo` 解决 `daemonic processes are not allowed to have children`
- [x] caption 去重 — marker 位图与矢量图 caption 相同则跳过
- [x] 扫描件退化 — 文本<100字符时 pymupdf 整页渲染为图片

### 前端功能
- [x] 知识库删除按钮 — hover 显示 + 确认弹窗 + 后端级联清理
- [x] 搜索结果详情弹窗 — 正确显示关联图片和 caption
- [x] 前端搜索组件 — 支持图片 nested 查询（images.caption）

### 后端修复
- [x] passlib → bcrypt 密码哈希修复
- [x] Python 文件语法错误修复（字面量 `\n`）
- [x] elasticsearch-py 9.x → 8.x 降级
- [x] ES 索引映射修复（移除 ik_smart）
- [x] Settings 模型 extra 字段兼容
- [x] markitdown[all] PDF 解析依赖安装
- [x] 搜索 API 端点实现
- [x] 索引名格式统一（kb_id → `_` 替换 `-`）
- [x] 应用层 RRF 融合（绕过 ES 付费许可证限制）
- [x] Celery event loop 复用修复
- [x] llama.cpp 自动安装（conda-forge）
