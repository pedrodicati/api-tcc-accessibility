import ollama
import io
import logging
from PIL import Image
from typing import List, Optional


class OllamaProcess:
    def __init__(self, model: str = "llama3.2-vision"):
        self.model = model
        ollama.pull(model)

    def make_prompt(
        self, prompt: str, image: Image.Image, system_prompt: Optional[str] = None
    ) -> List[dict]:
        # Converter imagem para bytes (PNG)
        byte_image = io.BytesIO()
        image.save(byte_image, format="PNG")
        byte_image = byte_image.getvalue()

        # Montar mensagens para o modelo
        messages = [
            {
                "role": "user",
                "content": prompt,
                "images": [byte_image],  # Verifique se o Ollama aceita 'images'
            }
        ]

        if system_prompt:
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": system_prompt,
                },
            )

        return messages

    def predict_model(self, prompt: List[dict]) -> str:
        try:
            response = ollama.chat(model=self.model, messages=prompt)

            return response.get("message", {}).get("content", "")
        except Exception as e:
            logging.error(f"Erro ao gerar a resposta: {e}")
            return "Erro ao gerar a resposta."
