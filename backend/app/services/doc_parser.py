"""
文档解析服务
PDF      → marker-pdf (本地 OCR + 布局分析)
Office   → markitdown (docx/pptx/xlsx)
MD / TXT → 直接读取
"""
import os
import io
import functools
from pathlib import Path


# ---------------------------------------------------------------------------
# 全局模型缓存：避免每次请求都重新加载模型
# ---------------------------------------------------------------------------
_model_cache: dict = {}


def _get_model_dict() -> dict:
    """获取或创建 marker 模型字典（带缓存）"""
    if not _model_cache:
        from marker.models import create_model_dict
        _model_cache.update(create_model_dict())
    return _model_cache


def _get_converter():
    """获取或创建 PdfConverter 实例（带缓存）"""
    if "_converter" not in _model_cache:
        from marker.converters.pdf import PdfConverter
        artifact_dict = _get_model_dict()
        _model_cache["_converter"] = PdfConverter(artifact_dict=artifact_dict)
    return _model_cache["_converter"]


class DocParserService:
    """文档解析 — 转 Markdown"""

    @staticmethod
    async def parse_to_markdown(file_path: str, file_type: str) -> tuple[str, list[dict], dict]:
        file_type = file_type.lower().lstrip(".")

        if file_type in ("md", "txt"):
            return await DocParserService._parse_text(file_path)
        elif file_type == "pdf":
            return await DocParserService._parse_pdf_with_marker(file_path)
        else:
            return await DocParserService._parse_with_markitdown(file_path, file_type)

    @staticmethod
    async def parse_pdf_fast(file_path: str) -> tuple[str, list[dict], dict]:
        """PDF 快速模式：marker fast 提文本+图片；扫描件退化为整页图片"""
        import asyncio
        from marker.config.parser import ConfigParser
        from marker.converters.pdf import PdfConverter
        from marker.output import MarkdownOutput
        import io

        def _extract_marker():
            cli_options: dict = {"mode": "fast", "output_format": "markdown"}
            config_parser = ConfigParser(cli_options)
            config = config_parser.generate_config_dict()
            if "_model_cache" not in globals():
                globals()["_model_cache"] = {}
            if not globals()["_model_cache"]:
                from marker.models import create_model_dict
                globals()["_model_cache"].update(create_model_dict())
            artifact_dict = globals()["_model_cache"]
            converter = PdfConverter(config=config, artifact_dict=artifact_dict)
            rendered: MarkdownOutput = converter(file_path)
            return rendered

        try:
            rendered = await asyncio.to_thread(_extract_marker)
        except Exception as e:
            if "llama-server" in str(e).lower() or "binary not found" in str(e).lower():
                print(f"[marker] OCR 不可用，退化为整页图片: {e}")
                return await DocParserService._parse_pdf_scanned(file_path)
            raise

        markdown_content = rendered.markdown or ""
        extracted_images = []

        if rendered.images:
            for img_name, pil_img in rendered.images.items():
                img_buffer = io.BytesIO()
                pil_img.save(img_buffer, format="PNG")
                img_data = img_buffer.getvalue()
                caption = DocParserService._find_image_caption(markdown_content, img_name)
                extracted_images.append({
                    "path": img_name, "data": img_data, "caption": caption, "page": 0,
                })

        # 扫描件退化：文本太少 → 每页渲染为一张图片
        if len(markdown_content) < 100:
            print(f"[marker] 文本太少({len(markdown_content)}字符)，退化为整页图片")
            return await DocParserService._parse_pdf_scanned(file_path)

        # 补充检测矢量图（marker 无法提取的 Path+Text 组合图）
        vector_images = await DocParserService._detect_vector_figures(file_path)
        if vector_images:
            # caption 去重：如果矢量图与已有图片 caption 相同，跳过
            existing_captions = {
                img["caption"].split(":")[0].strip().lower()
                for img in extracted_images if img.get("caption")
            }
            deduped = []
            for vimg in vector_images:
                vcap = vimg.get("caption", "")
                if vcap:
                    vkey = vcap.split(":")[0].strip().lower()
                    if vkey in existing_captions:
                        print(f"[marker] 跳过重复矢量图: {vkey}")
                        continue
                deduped.append(vimg)
            vector_images = deduped
            print(f"[marker] 检测到 {len(vector_images)} 张矢量图")
            extracted_images.extend(vector_images)

        print(f"marker fast 解析完成 [pdf]: {len(markdown_content)} 字符, {len(extracted_images)} 张图片")
        return markdown_content, extracted_images, {}

    @staticmethod
    async def _detect_vector_figures(file_path: str) -> list[dict]:
        """检测 PDF 中的矢量图（Path+Text 组合图），裁剪渲染为图片"""
        import asyncio
        import pymupdf

        def _detect():
            doc = pymupdf.open(file_path)
            vector_images = []
            img_counter = 0

            for page_num in range(len(doc)):
                page = doc[page_num]
                drawings = page.get_drawings()
                if not drawings:
                    continue

                # 过滤小元素（装饰线、边框 < 20pt），保留有面积的矢量元素
                big_rects = [
                    d["rect"] for d in drawings
                    if d["rect"].width > 20 and d["rect"].height > 20
                ]
                if len(big_rects) < 3:
                    continue  # 少量矢量元素不足以构成图

                # 贪心聚类：相邻矩形合并
                clusters = []
                for r in sorted(big_rects, key=lambda x: (x.y0, x.x0)):
                    placed = False
                    for i, c in enumerate(clusters):
                        if (r.x0 <= c.x1 + 30 and r.x1 >= c.x0 - 30 and
                                r.y0 <= c.y1 + 30 and r.y1 >= c.y0 - 30):
                            clusters[i] = pymupdf.Rect(
                                min(c.x0, r.x0), min(c.y0, r.y0),
                                max(c.x1, r.x1), max(c.y1, r.y1))
                            placed = True
                            break
                    if not placed:
                        clusters.append(r)

                for cluster in clusters:
                    # 过滤过小区域（图标、标注框）
                    if cluster.width < 80 or cluster.height < 40:
                        continue

                    # 略过与已有位图重叠的区域（避免重复提取）
                    page_images = page.get_images(full=True)
                    skip = False
                    for img in page_images:
                        xref = img[0]
                        rects = page.get_image_rects(xref)
                        if rects:
                            ir = rects[0]
                            overlap = (min(cluster.x1, ir.x1) - max(cluster.x0, ir.x0),
                                       min(cluster.y1, ir.y1) - max(cluster.y0, ir.y0))
                            if overlap[0] > 0 and overlap[1] > 0:
                                skip = True
                                break
                    if skip:
                        continue

                    # 扩大一点边缘，渲染为 PNG
                    pad = 5
                    clip = pymupdf.Rect(
                        max(0, cluster.x0 - pad), max(0, cluster.y0 - pad),
                        cluster.x1 + pad, cluster.y1 + pad)
                    mat = pymupdf.Matrix(3, 3)  # 3x 分辨率
                    pix = page.get_pixmap(matrix=mat, clip=clip)
                    img_data = pix.tobytes("png")

                    # 提取 caption：搜索 cluster 底部区域 ±100pt（可能在图内或图下）
                    caption = ""
                    import re
                    # 先在 cluster 底部附近找 Figure/Table 标题
                    search_bottom = max(0, cluster.y1 - 80)
                    clip_cap = pymupdf.Rect(cluster.x0, search_bottom,
                                            cluster.x1, cluster.y1 + 100)
                    cap_text = page.get_text("text", clip=clip_cap).strip()
                    # 逐行找 "Figure N:" / "Table N:" 开头的行
                    for line in cap_text.split("\n"):
                        line = line.strip()
                        if re.match(r"^(Figure|Table|Fig\.?)\s*\d+[:.]", line, re.IGNORECASE):
                            caption = line
                            break
                    # 没找到标题就用首行（如果有实质内容）
                    if not caption and cap_text:
                        first_line = cap_text.split("\n")[0].strip()
                        if len(first_line) > 15:
                            caption = first_line

                    vector_images.append({
                        "path": f"vector_fig_{img_counter}.png",
                        "data": img_data,
                        "caption": caption,
                        "page": page_num,
                    })
                    img_counter += 1
                    pix = None

            doc.close()
            return vector_images

        return await asyncio.to_thread(_detect)

    @staticmethod
    async def _parse_pdf_scanned(file_path: str) -> tuple[str, list[dict], dict]:
        """扫描件：每页渲染为一张图片"""
        import asyncio
        import pymupdf

        def _render():
            doc = pymupdf.open(file_path)
            images = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                mat = pymupdf.Matrix(2, 2)  # 2x 分辨率
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                images.append({
                    "path": f"page_{page_num}.png",
                    "data": img_data,
                    "caption": f"第 {page_num + 1} 页",
                    "page": page_num,
                })
                pix = None
            doc.close()
            return images

        images = await asyncio.to_thread(_render)
        print(f"扫描件渲染完成: {len(images)} 页")
        return "", images, {}

    @staticmethod
    async def _parse_text(file_path: str) -> tuple[str, list[dict], dict]:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read(), [], {}

    @staticmethod
    async def _parse_pdf_with_marker(file_path: str) -> tuple[str, list[dict], dict]:
        """使用 marker-pdf 解析 PDF"""
        import asyncio
        from marker.output import MarkdownOutput

        converter = _get_converter()

        # marker-pdf 的 __call__ 是同步的，放到线程中执行避免阻塞
        rendered: MarkdownOutput = await asyncio.to_thread(converter, file_path)

        markdown_content = rendered.markdown or ""
        extracted_images = []

        # rendered.images 是 {filename: PIL.Image} 字典
        if rendered.images:
            for img_name, pil_img in rendered.images.items():
                # 将 PIL Image 转为 bytes
                img_buffer = io.BytesIO()
                pil_img.save(img_buffer, format="PNG")
                img_data = img_buffer.getvalue()

                # 从 markdown 中查找 caption
                caption = DocParserService._find_image_caption(markdown_content, img_name)

                extracted_images.append({
                    "path": img_name,
                    "data": img_data,
                    "caption": caption,
                    "page": 0,
                })

        print(f"marker-pdf 解析完成 [pdf]: {len(markdown_content)} 字符, {len(extracted_images)} 张图片")
        return markdown_content, extracted_images, {}

    @staticmethod
    async def _parse_with_markitdown(file_path: str, file_type: str) -> tuple[str, list[dict], dict]:
        """使用 markitdown 解析 Office 文档"""
        import asyncio
        from markitdown import MarkItDown

        # markitdown convert 也是同步的，放到线程中
        def _convert():
            md = MarkItDown()
            return md.convert(file_path)

        result = await asyncio.to_thread(_convert)
        markdown_content = result.text_content or ""

        print(f"markitdown 解析完成 [{file_type}]: {len(markdown_content)} 字符")
        return markdown_content, [], {}

    @staticmethod
    async def _parse_pdf_pymupdf(file_path: str) -> tuple[str, list[dict], dict]:
        """pymupdf 快速提取：段落文本 + 图片 + caption（不跑 OCR）"""
        import asyncio
        import pymupdf

        def _extract():
            doc = pymupdf.open(file_path)
            page_count = len(doc)
            all_images = []
            markdown_parts = []

            for page_num in range(page_count):
                page = doc[page_num]
                blocks = page.get_text("dict")["blocks"]
                images_info = page.get_images(full=True)

                # 收集文本块和图片，按 y 坐标排序
                items = []

                # 文本块
                for block in blocks:
                    if block.get("type") == 0:  # text block
                        bbox = block["bbox"]
                        text_lines = []
                        for line in block.get("lines", []):
                            line_text = ""
                            for span in line.get("spans", []):
                                line_text += span.get("text", "")
                            if line_text.strip():
                                text_lines.append(line_text.strip())
                        text = " ".join(text_lines).strip()
                        if text:
                            items.append({
                                "y": bbox[1],
                                "type": "text",
                                "content": text,
                            })

                # 图片
                for img_idx, img in enumerate(images_info):
                    xref = img[0]
                    img_rects = page.get_image_rects(xref)
                    rect = img_rects[0] if img_rects else None
                    if rect:
                        pix = pymupdf.Pixmap(doc, xref)
                        if pix.n - pix.alpha > 3:
                            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                        img_data = pix.tobytes("png")
                        pix = None

                        # 找 caption：图片下方 80pt 内
                        clip = pymupdf.Rect(rect.x0, rect.y1, rect.x1, rect.y1 + 80)
                        caption = page.get_text("text", clip=clip).strip()

                        img_info = {
                            "path": f"image_{page_num}_{img_idx}.png",
                            "data": img_data,
                            "caption": caption,
                            "page": page_num,
                        }
                        all_images.append(img_info)
                        items.append({
                            "y": rect.y1,
                            "type": "image",
                            "index": len(all_images) - 1,
                        })

                # 按 y 坐标排序，生成 markdown
                items.sort(key=lambda x: x["y"])
                for item in items:
                    if item["type"] == "text":
                        markdown_parts.append(item["content"])
                    elif item["type"] == "image":
                        img = all_images[item["index"]]
                        alt = img["caption"] or img["path"]
                        markdown_parts.append(f"![{alt}]({img['path']})")

            doc.close()
            markdown_content = "\n\n".join(markdown_parts)
            return page_count, all_images, markdown_content

        page_count, images, markdown_content = await asyncio.to_thread(_extract)
        print(f"pymupdf 快速解析完成 [pdf]: {page_count} 页, {len(images)} 张图片, {len(markdown_content)} 字符")
        return markdown_content, images, {}

    @staticmethod
    def _find_image_caption(markdown: str, img_name: str) -> str:
        """从 markdown 中查找图片的 caption（支持 marker 格式）"""
        import re
        # 标准格式: ![caption](path)
        pattern = re.compile(r'!\[([^\]]*)\]\(([^)]*' + re.escape(img_name) + r'[^)]*)\)')
        match = pattern.search(markdown)
        if match and match.group(1):
            return match.group(1)
        # marker 格式: ![alt](path) 后跟 Figure X: caption
        if match:
            after = markdown[match.end():]
            lines = [l.strip() for l in after.split('\n') if l.strip()]
            if lines:
                first_line = lines[0]
                # 匹配 "Figure X: ..." 或 "Table X: ..."
                fig_match = re.match(r'(?:Figure|Table|Fig\.?)\s*\d+[:.\s]*(.*)', first_line, re.IGNORECASE)
                if fig_match:
                    return first_line
        return ""


async def parse_document(file_path: str, file_type: str, fast: bool = False) -> tuple[str, list[dict], dict]:
    if fast and file_type.lower().lstrip(".") == "pdf":
        return await DocParserService.parse_pdf_fast(file_path)
    return await DocParserService.parse_to_markdown(file_path, file_type)
