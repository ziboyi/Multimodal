# TODO List

## 📄 文档 (Docs)

- [ ] README 更新 — 补充本地开发启动流程（conda 环境 + docker 基础设施 + uvicorn + celery）
- [ ] API 文档 — 补充搜索接口 `/api/search` 的请求/响应格式说明
- [ ] 部署文档 — 补充 `--profile docker` 部署方式、elasticsearch-py 版本约束、markitdown[all] 依赖说明
- [ ] 环境变量文档 — 整理 `.env.example` 中所有配置项的说明（含新增的 `MINIO_CONSOLE_PORT`、`VITE_API_BASE_URL`）
- [ ] 故障排查文档 — 记录已知问题（passlib/bc 不兼容、ES 客户端版本、RRF 许可证限制）

---

## 🔧 开发 (Development)

### 用户管理
- [ ] 登录功能 — 前端登录页面与后端 `/api/auth/login` 对接
- [ ] 密码重置 — 支持通过邮箱验证重置密码
- [ ] 用户注册优化 — 邮箱验证、密码强度校验
- [ ] 用户资料 — 查看/编辑个人信息（full_name、avatar_url）
- [ ] Token 刷新 — 前端自动刷新 access_token 逻辑

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
- [ ] 异步处理状态推送 — WebSocket 实时通知文档处理进度
- [ ] 批量上传 — 支持同时上传多个文件
- [ ] 重复文档检测 — 基于文件哈希去重
- [ ] 嵌入模型预热 — Celery worker 启动时预加载模型到内存，避免每次请求重新加载（当前每次 ~25s）
- [ ] 索引批量优化 — bulk 索引批次大小调优

---

## 🧪 测试 (Testing)

### 多模态检索功能测试
- [ ] 文本检索测试 — 验证 hybrid/semantic/keyword 三种模式返回结果准确性
- [ ] 图像检索测试 — 上传含图片的 PDF，验证图片提取和图像-文本联合检索
- [ ] 跨语言检索测试 — 中英文混合文档的检索效果
- [ ] 检索相关性评估 — 构建测试集评估召回率和准确率
- [ ] 性能测试 — 大量 chunks 下的检索响应时间
- [ ] 并发测试 — 多用户同时检索的稳定性

---

## ✅ 已完成（本轮修复）

- [x] 后端本地开发启动（不依赖 Docker 打包）
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
- [x] 用户密码重置
