"""测试 PDF 解析功能（marker-pdf）"""
import asyncio
import sys
import time

sys.path.insert(0, "/home/zibo/桌面/Multimodal/backend")

from app.services.doc_parser import DocParserService


async def main(pdf_path: str):
    t0 = time.time()
    print(f"[*] 解析: {pdf_path}")
    markdown, images, page_info = await DocParserService.parse_to_markdown(pdf_path, "pdf")
    elapsed = time.time() - t0
    print(f"[+] 耗时: {elapsed:.1f}s")
    print(f"[+] Markdown 长度: {len(markdown)} 字符")
    print(f"[+] 提取图片: {len(images)} 张")
    print(f"[+] 页码信息: {page_info}")
    print("=" * 60)
    print("Markdown 前 1500 字符:")
    print(markdown[:1500])


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
