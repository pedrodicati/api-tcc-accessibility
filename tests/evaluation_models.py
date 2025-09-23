"""
Evaluation module for image captioning models.

This module provides functionality to evaluate image captioning models
using various metrics including BLEU, ROUGE, METEOR, and BERTScore.
"""

import io
import logging
import time
from itertools import islice
from typing import Tuple

import pandas as pd
import psutil
import requests
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

import nltk
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from transformers import BertTokenizer

logging.basicConfig(level=logging.INFO)

nltk.download("wordnet")
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")


class ImageCaptionEvaluator:
    """
    Evaluator for image captioning models using various NLP metrics.
    
    This class provides functionality to evaluate image captioning models
    against reference captions using BLEU, ROUGE, METEOR, and BERTScore metrics.
    
    Attributes:
    -----------
    api_url : str
        URL of the API endpoint for image captioning.
    num_samples : int
        Number of samples to evaluate.
    name_model : str
        Name of the model being evaluated.
    dataset : Dataset
        The dataset containing images and reference captions.
    results : List
        List to store evaluation results.
    """
    
    def __init__(self, api_url: str, num_samples: int = 10, name_model: str = "") -> None:
        """
        Initialize the evaluator with API URL and configuration.
        
        Parameters:
        -----------
        api_url : str
            URL of the image captioning API.
        num_samples : int
            Number of samples to evaluate.
        name_model : str
            Name of the model being evaluated.
        """
        self.api_url = api_url
        self.num_samples = num_samples
        self.name_model = name_model
        self.dataset = load_dataset(
            "laicsiifes/coco-captions-pt-br", split="train"
        ).shuffle(seed=42)
        self.results = []

    def get_cpu_gpu_usage(self) -> Tuple[float, float, float]:
        """
        Get current CPU and GPU usage statistics.
        
        Returns:
        --------
        Tuple[float, float, float]
            Tuple containing CPU usage %, memory usage %, and GPU memory in MB.
        """
        cpu_usage = psutil.cpu_percent()
        mem_usage = psutil.virtual_memory().percent
        gpu_mem = (
            torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
        )
        return cpu_usage, mem_usage, gpu_mem

    def evaluate(self) -> None:
        """
        Evaluate the image captioning model on a subset of the dataset.
        
        This method processes images from the dataset, generates captions,
        and computes various NLP metrics for evaluation.
        """
        rouge = Rouge()

        audio_path = "../assets/audio/audio-test2.m4a"
        with open(audio_path, "rb") as audio_file:
            audio_data = audio_file.read()

        for example in tqdm(
            islice(self.dataset, self.num_samples), total=self.num_samples
        ):
            image = Image.open(io.BytesIO(requests.get(example["url"]).content))
            img_bytes = self.image_to_bytes(image)
            reference_captions = example["caption"]

            cpu_before, mem_before, gpu_before = self.get_cpu_gpu_usage()
            start_time = time.perf_counter()

            generated_caption = self.get_generated_caption(
                img_bytes, io.BytesIO(audio_data)
            )

            end_time = time.perf_counter()
            cpu_after, mem_after, gpu_after = self.get_cpu_gpu_usage()

            latency = end_time - start_time
            cpu_usage = cpu_after - cpu_before
            mem_usage = mem_after - mem_before
            gpu_usage = gpu_after - gpu_before

            try:
                bleu_score_value = sentence_bleu(
                    [ref.split() for ref in reference_captions],
                    generated_caption.split(),
                )
                rouge_score = rouge.get_scores(
                    generated_caption, reference_captions[0]
                )[0]["rouge-l"]["f"]
                meteor = meteor_score([reference_captions], [generated_caption])
                precision, recall, bert_f1 = score(
                    [generated_caption],
                    [reference_captions[0]],
                    lang="pt",
                    device="cpu",
                )

                self.results.append(
                    {
                        "image_id": example["filename"],
                        "generated_caption": generated_caption,
                        "reference_captions": reference_captions,
                        "bleu": bleu_score_value,
                        "rouge_l": rouge_score,
                        "meteor": meteor,
                        "bert_score_f1": bert_f1.item(),
                        "latency": latency,
                        "cpu_usage": cpu_usage,
                        "mem_usage": mem_usage,
                        "gpu_usage": gpu_usage,
                    }
                )
            except Exception as exc:
                logging.error(
                    "Erro ao processar imagem %s: %s", 
                    example['filename'], str(exc)
                )

        self.save_results()

    def image_to_bytes(self, image: Image.Image) -> bytes:
        """
        Convert PIL Image to bytes in JPEG format.
        
        Parameters:
        -----------
        image : Image.Image
            PIL Image object to convert.
            
        Returns:
        --------
        bytes
            Image data as bytes in JPEG format.
        """
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return buffer.getvalue()

    def get_generated_caption(self, img_bytes: bytes, audio: io.BytesIO) -> str:
        """
        Get caption from API for given image and audio.
        
        Parameters:
        -----------
        img_bytes : bytes
            Image data as bytes.
        audio : io.BytesIO
            Audio data as BytesIO object.
            
        Returns:
        --------
        str
            Generated caption from the API.
        """
        files = {
            "image": ("image.jpg", img_bytes, "image/jpeg"),
            "audio": ("audio.m4a", audio, "audio/m4a"),
        }
        try:
            response = requests.post(self.api_url, files=files)
            response.raise_for_status()
            response_json = response.json()
            return response_json.get("image_text", "")
        except requests.exceptions.Timeout:
            logging.warning("Timeout ao chamar a API.")
            return ""
        except requests.exceptions.RequestException as exc:
            logging.error("Erro na requisição da API: %s", str(exc))
            return ""

    def save_results(self) -> None:
        """
        Save evaluation results to a CSV file.
        
        The results are saved to a CSV file named with the model name.
        """
        data_frame = pd.DataFrame(self.results)
        filename = f"evaluation_results_{self.name_model}.csv"
        data_frame.to_csv(filename, index=False)
        logging.info("Resultados salvos em %s", filename)


evaluator = ImageCaptionEvaluator(
    api_url="http://localhost:3015/api/analyze-image-audio-query",
    num_samples=1000,
    name_model="Llama-3.2-11B-Vision-Instruct",
)
evaluator.evaluate()
