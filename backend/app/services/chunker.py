"""
文本分块服务
支持多种分块策略 + 图片-文本关联（基于页码）
"""
from typing import List, Optional
from langdetect import detect
import re


class TextChunk:
    """文本块"""
    def __init__(self, content: str, chunk_index: int, chunk_type: str = "text",
                 page_number: int | None = None, metadata: dict | None = None):
        self.content = content
        self.chunk_index = chunk_index
        self.chunk_type = chunk_type
        self.page_number = page_number
        self.metadata = metadata or {}


class ChunkerService:
    """文本分块服务"""

    @staticmethod
    def chunk_by_paragraph(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[TextChunk]:
        """按段落分块"""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current_content = ""
        chunk_index = 0
        for para in paragraphs:
            if len(current_content) + len(para) <= chunk_size:
                current_content += ("\n\n" + para) if current_content else para
            else:
                if current_content:
                    chunks.append(TextChunk(content=current_content, chunk_index=chunk_index, chunk_type="text"))
                    chunk_index += 1
                    overlap_text = current_content[-chunk_overlap:] if chunk_overlap > 0 else ""
                    current_content = overlap_text + "\n\n" + para if overlap_text else para
        if current_content:
            chunks.append(TextChunk(content=current_content, chunk_index=chunk_index, chunk_type="text"))
        return chunks

    @staticmethod
    def chunk_by_fixed_size(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[TextChunk]:
        """固定长度分块"""
        chunks = []
        start = 0
        chunk_index = 0
        text_length = len(text)
        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunks.append(TextChunk(content=text[start:end], chunk_index=chunk_index, chunk_type="text"))
            chunk_index += 1
            start += chunk_size - chunk_overlap
        return chunks

    @staticmethod
    def chunk_markdown(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[TextChunk]:
        """Markdown 感知的分块"""
        parts = re.split(r'(#{1,6}\s+.*|\n{2,})', text)
        parts = [p.strip() for p in parts if p and p.strip()]

        chunks = []
        current_content = ""
        chunk_index = 0
        chunk_type = "text"

        for part in parts:
            if re.match(r'^#{1,6}\s+', part):
                if current_content:
                    chunks.append(TextChunk(content=current_content, chunk_index=chunk_index, chunk_type=chunk_type))
                    chunk_index += 1
                    overlap_text = current_content[-chunk_overlap:] if chunk_overlap > 0 else ""
                    current_content = overlap_text
                    chunk_type = "text"
                current_content += ("\n\n" + part) if current_content else part
                chunk_type = "title"
            else:
                if len(current_content) + len(part) <= chunk_size:
                    current_content += ("\n\n" + part) if current_content else part
                else:
                    if current_content:
                        chunks.append(TextChunk(content=current_content, chunk_index=chunk_index, chunk_type=chunk_type))
                        chunk_index += 1
                        overlap_text = current_content[-chunk_overlap:] if chunk_overlap > 0 else ""
                        current_content = overlap_text + "\n\n" + part if overlap_text else part
                    else:
                        for sub in ChunkerService.chunk_by_fixed_size(part, chunk_size, chunk_overlap):
                            sub.chunk_index = chunk_index
                            chunks.append(sub)
                            chunk_index += 1
                        current_content = ""

        if current_content:
            chunks.append(TextChunk(content=current_content, chunk_index=chunk_index, chunk_type=chunk_type))

        return chunks

    @staticmethod
    def assign_page_numbers(chunks: List[TextChunk], page_info: dict, full_text_length: int) -> None:
        """根据页面信息为 chunk 分配页码"""
        page_lengths = page_info.get("page_lengths", {})
        if not page_lengths or full_text_length == 0:
            return

        total_units = sum(page_lengths.values())
        if total_units == 0:
            return

        # 计算每页在全文中的字符偏移范围
        page_ranges = {}
        current_offset = 0
        max_page = max(page_lengths.keys())
        for page_id in range(max_page + 1):
            page_units = page_lengths.get(page_id, 0)
            estimated_chars = int((page_units / total_units) * full_text_length)
            page_ranges[page_id] = (current_offset, current_offset + estimated_chars)
            current_offset += estimated_chars

        # 按 chunk 偏移量分配页码
        chunk_offset = 0
        for chunk in chunks:
            chunk_mid = chunk_offset + len(chunk.content) // 2
            chunk_offset += len(chunk.content)

            assigned = False
            for page_id, (start, end) in page_ranges.items():
                if start <= chunk_mid < end:
                    chunk.page_number = page_id + 1
                    assigned = True
                    break

            if not assigned:
                chunk.page_number = max_page + 1

    @staticmethod
    def associate_images_with_chunks(chunks: List[TextChunk], images: list[dict]) -> list[dict]:
        """将图片与最相关的 chunks 关联"""
        figure_pattern = re.compile(r'(?:Figure|Fig\.?)\s*(\d+)', re.IGNORECASE)

        result_images = []
        for idx, img in enumerate(images):
            img_page = img.get("page", 0)
            best_chunk_idx = None
            best_score = -1

            for chunk_idx, chunk in enumerate(chunks):
                # 1. 直接引用
                matches = figure_pattern.findall(chunk.content)
                for match in matches:
                    if int(match) == idx + 1:
                        score = 100
                        if score > best_score:
                            best_score = score
                            best_chunk_idx = chunk_idx

                # 2. 页码匹配
                if chunk.page_number and img_page > 0:
                    page_diff = abs(chunk.page_number - img_page)
                    score = max(0, 50 - page_diff * 20)
                    if score > best_score:
                        best_score = score
                        best_chunk_idx = chunk_idx

            result_images.append({
                **img,
                "associated_chunk_index": best_chunk_idx if best_chunk_idx is not None else min(idx, len(chunks) - 1),
            })

        return result_images

    @staticmethod
    def detect_language(text: str) -> str:
        try:
            return detect(text)
        except Exception:
            return "unknown"


async def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200,
                     strategy: str = "markdown") -> List[TextChunk]:
    if strategy == "paragraph":
        return ChunkerService.chunk_by_paragraph(text, chunk_size, chunk_overlap)
    elif strategy == "fixed":
        return ChunkerService.chunk_by_fixed_size(text, chunk_size, chunk_overlap)
    else:
        return ChunkerService.chunk_markdown(text, chunk_size, chunk_overlap)
