import time
import torch
import psutil
import logging
import pandas as pd
import requests
import io
import nltk
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
from transformers import BertTokenizer
from itertools import islice

logging.basicConfig(level=logging.INFO)

nltk.download("wordnet")
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")


class ImageCaptionEvaluator:
    def __init__(self, api_url: str, num_samples: int = 10, name_model: str = ""):
        self.api_url = api_url
        self.num_samples = num_samples
        self.name_model = name_model
        self.dataset = load_dataset(
            "laicsiifes/coco-captions-pt-br", split="train"
        ).shuffle(seed=42)
        self.results = []

    def get_cpu_gpu_usage(self):
        cpu_usage = psutil.cpu_percent()
        mem_usage = psutil.virtual_memory().percent
        gpu_mem = (
            torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
        )
        return cpu_usage, mem_usage, gpu_mem

    def evaluate(self):
        rouge = Rouge()
        device = "cuda" if torch.cuda.is_available() else "cpu"

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
                P, R, F1 = score(
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
                        "bert_score_f1": F1.item(),
                        "latency": latency,
                        "cpu_usage": cpu_usage,
                        "mem_usage": mem_usage,
                        "gpu_usage": gpu_usage,
                    }
                )
            except Exception as e:
                print(f"❌ Erro ao calcular métricas: {e}")

        self.save_results()

    def image_to_bytes(self, image):
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return buffer.getvalue()

    def get_generated_caption(self, img_bytes, audio):
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
            print("⏱️ Timeout ao chamar a API.")
            return ""
        except requests.exceptions.RequestException as e:
            print(f"❌ Erro na requisição da API: {e}")
            return ""

    def save_results(self):
        df = pd.DataFrame(self.results)
        df.to_csv(f"evaluation_results_{self.name_model}.csv", index=False)
        print(f"✅ Resultados salvos em evaluation_results_{self.name_model}.csv")


evaluator = ImageCaptionEvaluator(
    api_url="http://localhost:3015/api/analyze-image-audio-query",
    num_samples=1000,
    name_model="Llama-3.2-11B-Vision-Instruct",
)
evaluator.evaluate()
