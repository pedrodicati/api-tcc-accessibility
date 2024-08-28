import torch
import logging
import warnings
import re
import io
import os

from PIL import Image
from transformers import pipeline
from transformers import BitsAndBytesConfig
from typing import Union, List, Dict

class ImageProcess:
    def __init__(self, model_id: Union[str, None] = None):
        if not model_id:
            warnings.warn("No model id provided, using the default one")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        if device == torch.device("cuda"):
            logging.info("Using GPU")

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
            )
        else:
            logging.info("Using CPU")

        try:
            self.pipe = pipeline(
                "image-to-text",
                model=model_id,
                device=device,
                model_kwargs={
                    "quantization_config": quantization_config,
                }
                if device == torch.device("cuda")
                else None,
            )
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise Exception("Error loading model")

        logging.info("Model to describe images loaded successfully.")

    def make_input_prompt(self, question: str) -> str:
        return f"USER: <image>\n{question}\nASSISTANT:"
    
    def process_output_text(self, output: str) -> str:
        pattern = r'USER:.*?ASSISTANT: '

        return re.sub(pattern, "", output, flags=re.DOTALL).strip()

    def image_to_text(
        self,
        image_or_file: Union[str, bytes, None] = None,
        question: Union[str, None] = None,
    ) -> List[Dict[str, str]]:
        if not image_or_file or not question:
            raise ValueError("No image or question provided. Please provide both.")

        if isinstance(image_or_file, str):
            if not os.path.exists(image_or_file):
                raise FileNotFoundError(f"File not found: {image_or_file}")

            image = Image.open(image_or_file)
        else:
            image = Image.open(io.BytesIO(image_or_file))

        if image.size[0] == 0 or image.size[1] == 0:
            raise ValueError("The input image has no tokens.")

        prompt = self.make_input_prompt(question)

        generated_text = self.pipe(image, prompt=prompt, max_new_tokens=1000)[0].get("generated_text")

        return self.process_output_text(generated_text)
