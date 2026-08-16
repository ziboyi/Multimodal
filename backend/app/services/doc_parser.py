"""
文档解析服务
PDF → Marker，其他 → markitdown
"""
from typing import Optional
from pathlib import Path
import tempfile


class DocParserService:
    """文档解析 — 统一转 Markdown"""

    @staticmethod
    async def parse_to_markdown(file_path: str, file_type: str) -> tuple[str, list[dict]]:
        """
        解析文件为 Markdown
        返回: (markdown_content, extracted_images)
        extracted_images: [{"path": str, "data": bytes, "caption": str}]
        """
        file_type = file_type.lower().lstrip(".")

        if file_type == "pdf":
            return await DocParserService._parse_pdf(file_path)
        elif file_type in ("md", "txt"):
            return await DocParserService._parse_text(file_path)
        else:
            return await DocParserService._parse_with_markitdown(file_path, file_type)

    @staticmethod
    async def _parse_pdf(file_path: str) -> tuple[str, list[dict]]:
        """PDF 用 Marker 解析"""
        try:
            from marker.convert import convert_single_pdf
            from marker.models import load_all_models

            # 加载模型（首次调用时缓存）
            model_lst = load_all_models()

            # 转换
            full_text, images, out_meta = convert_single_pdf(
                file_path,
                model_lst,
                max_pages=None,
                langs=None,
                batch_multiplier=2,
            )

            # images 是 {path: PIL.Image} 格式
            extracted_images = []
            for path, img in images.items():
                import io
                img_bytes = io.BytesIO()
                img.save(img_bytes, format="PNG")
                extracted_images.append({
                    "path": path,
                    "data": img_bytes.getvalue(),
                    "caption": "",
                })

            return full_text, extracted_images
        except ImportError:
            # Marker 未安装，降级到 markitdown
            return await DocParserService._parse_with_markitdown(file_path, "pdf")

    @staticmethod
    async def _parse_text(file_path: str) -> tuple[str, list[dict]]:
        """纯文本文件直接读取"""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read(), []

    @staticmethod
    async def _parse_with_markitdown(file_path: str, file_type: str) -> tuple[str, list[dict]]:
        """其他格式用 markitdown"""
        from markitdown import MarkItDown

        md = MarkItDown()
        result = md.convert(file_path)
        return result.text_content, []


# 便捷函数
async def parse_document(file_path: str, file_type: str) -> tuple[str, list[dict]]:
    """解析文档为 Markdown"""
    return await DocParserService.parse_to_markdown(file_path, file_type)
