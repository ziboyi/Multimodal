"""
文本分块服务
支持多种分块策略
"""
from typing import List, Optional
from langdetect import detect


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
        """按段落分块，超过 chunk_size 时合并小段落或切分大段落"""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current_content = ""
        chunk_index = 0

        for para in paragraphs:
            if len(current_content) + len(para) <= chunk_size:
                current_content += ("\n\n" + para) if current_content else para
            else:
                if current_content:
                    chunks.append(TextChunk(
                        content=current_content,
                        chunk_index=chunk_index,
                        chunk_type="text"
                    ))
                    chunk_index += 1
                    # 重叠
                    overlap_text = current_content[-chunk_overlap:] if chunk_overlap > 0 else ""
                    current_content = overlap_text + "\n\n" + para if overlap_text else para

        if current_content:
            chunks.append(TextChunk(
                content=current_content,
                chunk_index=chunk_index,
                chunk_type="text"
            ))

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
            chunk_text = text[start:end]
            chunks.append(TextChunk(
                content=chunk_text,
                chunk_index=chunk_index,
                chunk_type="text"
            ))
            chunk_index += 1
            start += chunk_size - chunk_overlap

        return chunks

    @staticmethod
    def chunk_markdown(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[TextChunk]:
        """Markdown 感知的分块 - 优先在标题、段落边界切分"""
        import re

        # 按标题和段落分割
        parts = re.split(r'(#{1,6}\s+.*|\n{2,})', text)
        parts = [p.strip() for p in parts if p and p.strip()]

        chunks = []
        current_content = ""
        chunk_index = 0
        chunk_type = "text"

        for part in parts:
            # 检测标题
            if re.match(r'^#{1,6}\s+', part):
                # 保存当前 chunk
                if current_content:
                    chunks.append(TextChunk(
                        content=current_content,
                        chunk_index=chunk_index,
                        chunk_type=chunk_type
                    ))
                    chunk_index += 1
                    overlap_text = current_content[-chunk_overlap:] if chunk_overlap > 0 else ""
                    current_content = overlap_text
                    chunk_type = "text"

                # 标题开始新 chunk
                current_content += ("\n\n" + part) if current_content else part
                chunk_type = "title"
            else:
                if len(current_content) + len(part) <= chunk_size:
                    current_content += ("\n\n" + part) if current_content else part
                else:
                    if current_content:
                        chunks.append(TextChunk(
                            content=current_content,
                            chunk_index=chunk_index,
                            chunk_type=chunk_type
                        ))
                        chunk_index += 1
                        overlap_text = current_content[-chunk_overlap:] if chunk_overlap > 0 else ""
                        current_content = overlap_text + "\n\n" + part if overlap_text else part
                    else:
                        # 单个段落超长，用固定长度切
                        for sub in ChunkerService.chunk_by_fixed_size(part, chunk_size, chunk_overlap):
                            sub.chunk_index = chunk_index
                            chunks.append(sub)
                            chunk_index += 1
                        current_content = ""

        if current_content:
            chunks.append(TextChunk(
                content=current_content,
                chunk_index=chunk_index,
                chunk_type=chunk_type
            ))

        return chunks

    @staticmethod
    def detect_language(text: str) -> str:
        """检测文本语言"""
        try:
            return detect(text)
        except Exception:
            return "unknown"


# 便捷函数
async def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200,
                     strategy: str = "markdown") -> List[TextChunk]:
    """分块入口"""
    if strategy == "paragraph":
        return ChunkerService.chunk_by_paragraph(text, chunk_size, chunk_overlap)
    elif strategy == "fixed":
        return ChunkerService.chunk_by_fixed_size(text, chunk_size, chunk_overlap)
    else:
        return ChunkerService.chunk_markdown(text, chunk_size, chunk_overlap)