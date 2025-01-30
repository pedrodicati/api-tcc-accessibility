import torch
import logging
import re
import io
import os
import gc
import warnings
import traceback

from PIL import Image
from transformers import pipeline, BitsAndBytesConfig
from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    MllamaForConditionalGeneration,
    AutoModelForImageTextToText,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info

from typing import Union, List, Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO)


class ImageProcess:
    """
    Classe responsável por gerenciar pipelines de imagem-para-texto usando
    diferentes modelos do Hugging Face Transformers. Permite trocar de modelo
    de forma flexível, mantendo-os em cache.

    Atributos:
    -----------
    models : Dict[str, pipeline]
        Dicionário que mapeia model_id -> instância de pipeline carregada.
    default_model_id : str
        ID de modelo padrão (usado se nenhum outro for informado).
    device : torch.device
        Dispositivo em que as inferências serão realizadas (CPU ou CUDA).
    torch_dtype : torch.dtype
        Tipo de dado utilizado nos tensores (float16, float32, etc.).
    """

    def __init__(
        self,
        model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        model_type: str = "image-text-to-text",
    ) -> None:
        """
        Inicializa a classe com um modelo padrão. Para já iniciar
        o processo com o modelo carregado, chamamos load_model().
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = (
            torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )

        self.model_id = model_id
        self.model_type = model_type
        self.models = {}

        logging.info(
            f"ImageProcess iniciado. Device: {self.device}, torch_dtype: {self.torch_dtype}"
        )
        logging.info(f"Modelo padrão configurado para: {self.model_id}")

        self.load_model(model_id)

    def load_model(
        self, model_id: str, model_type: Optional[str] = "image-text-to-text"
    ) -> None:
        """
        Carrega o pipeline correspondente ao model_id e armazena em self.models.
        Se já estiver carregado, apenas retorna o pipeline existente.

        Parâmetros:
        -----------
        model_id : str
            Identificador do modelo no Hugging Face (ex.: 'llava-hf/llava-v1.6-mistral-7b-hf').

        Retorna:
        --------
        pipeline
            Instância de pipeline carregada pronta para inferência.
        """
        if model_id in self.models:
            logging.info(
                f"Modelo '{model_id}' já está carregado. Retornando instância existente."
            )
            return self.models[model_id]

        # Se não estiver carregado, criar uma nova pipeline
        logging.info(f"Carregando novo modelo: {model_id}")

        quantization_config = None
        # Se estivermos em GPU, podemos configurar BitsAndBytes
        if self.device == torch.device("cuda"):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.torch_dtype,
                bnb_4bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            pass
        try:
            model_type = model_type or self.model_type

            if self.model_id == "llava-hf/llava-v1.6-mistral-7b-hf":
                self.processor = LlavaNextProcessor.from_pretrained(
                    self.model_id,
                    quantization_config=quantization_config,
                    device_map="auto",
                )

                self.model = LlavaNextForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    quantization_config=quantization_config,
                    device_map="auto",
                )
                self.model.to(self.device)

            elif self.model_id == "alpindale/Llama-3.2-11B-Vision-Instruct":
                self.processor = AutoProcessor.from_pretrained(self.model_id)

                self.model = MllamaForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=self.torch_dtype,
                    device_map="auto",
                    quantization_config=quantization_config,
                    low_cpu_mem_usage=True,
                )

            elif self.model_id == "Qwen/Qwen2.5-VL-7B-Instruct":
                self.processor = AutoProcessor.from_pretrained(self.model_id)

                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=self.torch_dtype,
                    device_map="auto",
                    quantization_config=quantization_config,
                    low_cpu_mem_usage=True,
                    # max_memory={0: "5GiB"}
                )

            else:
                warnings.warn(
                    f"ImageProcess não suporta o modelo '{model_id}'. Usando carregamento genérico.",
                    category=UserWarning,
                )

                self.processor = AutoProcessor.from_pretrained(
                    self.model_id,
                    quantization_config=quantization_config,
                    device_map="auto",
                    use_fast=True,
                )

                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_id,
                    torch_dtype=self.torch_dtype,
                    quantization_config=quantization_config,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )

        except Exception as e:
            logging.error(f"Erro ao carregar o modelo '{model_id}': {e}")
            raise RuntimeError(f"Falha ao carregar o modelo {model_id}. Erro: {str(e)}")

    def make_input_prompt(
        self, question: str, image: Image.Image
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Gera o prompt que será usado para o modelo, combinando a marcação <image>
        com a pergunta do usuário.

        Parâmetros:
        -----------
        question : str
            Pergunta do usuário (ex.: "O que há na imagem?").

        Retorna:
        --------
        str
            Prompt formatado para ser concatenado à imagem.
        """

        system_prompt = (
            "Você é um assistente virtual projetado para auxiliar pessoas com deficiência visual "
            "a compreenderem o ambiente ao seu redor, de forma clara e objetiva. A imagem fornecida "
            "representa exatamente o que um usuário veria se pudesse enxergar. Sua tarefa é descrever "
            "a cena de forma clara, precisa e acessível, respondendo também à pergunta feita pelo usuário. "
            "Seja detalhado o suficiente para que ele possa construir uma imagem mental do que está ao seu redor, "
            "porém evite descrições muito longas. Sinalize eventuais riscos e o que o usuário deve fazer para "
            "evitá-los. Seja sempre educado e prestativo. Vamos lá!"
        )

        chat = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            },
        ]

        print(chat)
        if self.model_id == "Qwen/Qwen2.5-VL-7B-Instruct":
            prompt = self.processor.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = self.processor.apply_chat_template(
                chat, add_generation_prompt=True
            )

        print(prompt)

        return prompt, chat

    def process_output_text(self, output: str) -> str:
        """
        Remove partes desnecessárias do texto gerado pelo modelo, incluindo
        tags de sistema e formatações extras.
        
        Parâmetros:
        -----------
        output : str
            Texto gerado pelo modelo.
        
        Retorna:
        --------
        str
            Texto limpo e pronto para exibição.
        """
        output = re.sub(r"<<SYS>>.*?<</SYS>>", "", output, flags=re.DOTALL).strip()
        output = re.sub(r"\[INST\].*?\[/INST\]", "", output, flags=re.DOTALL).strip()
        output = re.sub(r"\s+([.,!?])", r"\1", output)
        output = output.replace("\n", " ").strip()

        return output

    def check_image(self, image_or_file: Union[str, bytes]) -> Image.Image:
        # Verifica se é string (caminho) ou bytes (arquivo binário)
        if isinstance(image_or_file, str):
            if not os.path.exists(image_or_file):
                raise FileNotFoundError(
                    f"Arquivo de imagem não encontrado: {image_or_file}"
                )
            image = Image.open(image_or_file)
        else:
            image = Image.open(io.BytesIO(image_or_file))

        if image.size[0] == 0 or image.size[1] == 0:
            raise ValueError("A imagem fornecida parece inválida ou vazia.")

        return image

    def image_to_text(
        self,
        image_or_file: Union[str, bytes],
        question: str,
        max_new_tokens: int = 512,
    ) -> str:
        """
        Função principal para gerar uma descrição de imagem, dadas a imagem, a
        pergunta e um modelo específico (ou o modelo padrão).

        Parâmetros:
        -----------
        image_or_file : str ou bytes
            Caminho para o arquivo de imagem ou bytes da imagem.
        question : str
            Pergunta do usuário, que será combinada com o prompt padrão.
        model_id : str, opcional
            Identificador do modelo desejado. Se None, utiliza o modelo padrão.
        max_new_tokens : int
            Número máximo de tokens a serem gerados na saída.

        Retorna:
        --------
        str
            Texto final de descrição ou resposta gerada pelo modelo.
        """
        if not question:
            raise ValueError(
                "Nenhuma pergunta fornecida. 'question' não pode ser vazio."
            )

        image = self.check_image(image_or_file)

        # Monta o prompt
        prompt, chat = self.make_input_prompt(question, image)
        logging.info(f"Realizando inferência com o modelo '{self.model_id}'...")

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        try:
            if self.model_id == "llava-hf/llava-v1.6-mistral-7b-hf":
                inputs = self.processor(
                    text=prompt, images=image, return_tensors="pt"
                ).to(self.model.device)

                with torch.no_grad():
                    generate_ids = self.model.generate(
                        **inputs, max_new_tokens=max_new_tokens
                    )
                    generated_text = self.processor.decode(
                        generate_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )

            elif self.model_id == "Qwen/Qwen2.5-VL-7B-Instruct":
                image_inputs, video_inputs = process_vision_info(chat)

                inputs = self.processor(
                    text=prompt,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(self.model.device)
                
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs, max_new_tokens=max_new_tokens
                    )
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :]
                        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    generated_text = self.processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0]

            elif self.model_id == "alpindale/Llama-3.2-11B-Vision-Instruct":
                inputs = self.processor(
                    image, prompt, add_special_tokens=False, return_tensors="pt"
                ).to(self.model.device)

                output = self.model.generate(**inputs, max_new_tokens=30)
                generated_text = self.processor.decode(
                    output[0], skip_special_tokens=True
                )

            else:
                warnings.warn(
                    f"Modelo '{self.model_id}' não possui inferência implementada. Usando inferência genérica.",
                    category=UserWarning,
                )

                inputs = self.processor(
                    text=prompt, images=image, return_tensors="pt"
                ).to(self.model.device)

                with torch.no_grad():
                    generate_ids = self.model.generate(
                        **inputs, max_new_tokens=max_new_tokens
                    )
                    generated_text = self.processor.decode(
                        generate_ids[0], skip_special_tokens=True
                    )

            return self.process_output_text(generated_text)
        except Exception:
            logging.error(f"Erro durante inferência: {traceback.format_exc()}")
            raise RuntimeError(
                f"Falha na geração de texto a partir da imagem. Erro: {traceback.format_exc()}"
            )
