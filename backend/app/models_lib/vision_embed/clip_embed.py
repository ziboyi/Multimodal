import asyncio
from PIL import Image
import io
from transformers import CLIPProcessor, CLIPModel
from app.models_lib.base import VisionEmbedProvider


class CLIPEmbed(VisionEmbedProvider):
    """CLIP 视觉嵌入实现（本地模型）"""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self._model_name = model_name
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self.model.config.projection_dim

    async def embed(self, images: list[bytes]) -> list[list[float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._encode, images)

    def _encode(self, images: list[bytes]) -> list[list[float]]:
        pil_images = [Image.open(io.BytesIO(img)).convert("RGB") for img in images]
        inputs = self.processor(images=pil_images, return_tensors="pt")
        outputs = self.model.get_image_features(**inputs)
        # 归一化
        import torch.nn.functional as F
        embeddings = F.normalize(outputs, dim=-1)
        return embeddings.tolist()
