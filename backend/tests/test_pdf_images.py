"""
PDF 图片提取（跳过 OCR，用 pymupdf 直接提取）
"""
import asyncio
import sys
import time

sys.path.insert(0, "/home/zibo/桌面/Multimodal/backend")


async def extract_images_no_ocr(pdf_path: str):
    """纯 pymupdf 提取图片 + 附件 caption，不跑 OCR"""
    import fitz

    t0 = time.time()
    doc = fitz.open(pdf_path)
    page_count = len(doc)
    images = []
    for page_num in range(page_count):
        page = doc[page_num]
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n - pix.alpha > 3:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            img_data = pix.tobytes("png")
            img_rects = page.get_image_rects(xref)
            rect = img_rects[0] if img_rects else None
            caption = ""
            if rect:
                clip = fitz.Rect(rect.x0, rect.y1, rect.x1, rect.y1 + 80)
                caption = page.get_text("text", clip=clip).strip()
            images.append({
                "page": page_num + 1,
                "xref": xref,
                "data": img_data,
                "caption": caption,
                "width": pix.width,
                "height": pix.height,
            })
            pix = None
    doc.close()
    elapsed = time.time() - t0
    print(f"[+] 耗时: {elapsed:.2f}s")
    print(f"[+] 总页数: {page_count}")
    print(f"[+] 提取图片: {len(images)} 张")
    for i, img in enumerate(images):
        print(f"    [{i}] page={img['page']} {img['width']}x{img['height']} "
              f"{len(img['data'])}B caption='{img['caption'][:60]}'")
    return images


if __name__ == "__main__":
    asyncio.run(extract_images_no_ocr(sys.argv[1]))
