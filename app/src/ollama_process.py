"""
Ollama process module for vision models.

This module provides functionality to interact with Ollama models
for image-to-text processing, particularly for vision models.
"""

import io
import logging
from typing import Dict, List, Optional

import ollama
from PIL import Image


class OllamaProcess:
    """
    Ollama process handler for vision models.
    
    This class provides an interface to interact with Ollama vision models
    for image-to-text processing tasks.
    
    Attributes:
    -----------
    model : str
        The name of the Ollama model to use.
    """
    
    def __init__(self, model: str = "llama3.2-vision") -> None:
        """
        Initialize the Ollama process with a specified model.
        
        Parameters:
        -----------
        model : str
            The name of the Ollama model to use for inference.
        """
        self.model = model
        ollama.pull(model)

    def make_prompt(
        self, prompt: str, image: Image.Image, system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Create a formatted prompt for the Ollama model with image and text.
        
        Parameters:
        -----------
        prompt : str
            The text prompt to send to the model.
        image : Image.Image
            The PIL Image to include in the prompt.
        system_prompt : Optional[str]
            Optional system prompt to provide context.
            
        Returns:
        --------
        List[Dict[str, str]]
            Formatted messages ready for the Ollama model.
        """
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

    def predict_model(self, prompt: List[Dict[str, str]]) -> str:
        """
        Generate a response from the Ollama model using the formatted prompt.
        
        Parameters:
        -----------
        prompt : List[Dict[str, str]]
            The formatted messages for the model.
            
        Returns:
        --------
        str
            The generated response from the model.
        """
        try:
            response = ollama.chat(model=self.model, messages=prompt)

            return response.get("message", {}).get("content", "")
        except Exception as exc:
            logging.error("Erro ao gerar a resposta: %s", str(exc))
            return "Erro ao gerar a resposta."
