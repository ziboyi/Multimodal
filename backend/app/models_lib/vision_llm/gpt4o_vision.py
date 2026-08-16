import base64
from openai import AsyncOpenAI
from app.models_lib.base import VisionLLMProvider


class GPT4oVision(VisionLLMProvider):
    """GPT-4o 多模态视觉模型实现"""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str = "",
        api_base: str | None = None,
    ):
        self._model = model
        self.client = AsyncOpenAI(api_key=api_key, base_url=api_base)

    @property
    def model_name(self) -> str:
        return self._model

    async def describe(self, image: bytes, prompt: str = "描述这张图片") -> str:
        b64 = base64.b64encode(image).decode()
        response = await self.client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}"
                            },
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content or ""
