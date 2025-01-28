import torch
import logging
import warnings
import re
import io
import os

from PIL import Image
from transformers import pipeline, BitsAndBytesConfig
from typing import Union, List, Dict, Optional

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

    def __init__(self, default_model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf"):
        """
        Inicializa a classe com um modelo padrão. Não carrega nada neste
        momento; o carregamento real ocorre quando chamamos load_model().
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.default_model_id = default_model_id
        self.models: Dict[str, pipeline] = {}

        logging.info(
            f"ImageProcess iniciado. Device: {self.device}, torch_dtype: {self.torch_dtype}"
        )
        logging.info(f"Modelo padrão configurado para: {self.default_model_id}")

        self.load_model(default_model_id)

    def load_model(self, model_id: str) -> pipeline:
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
                bnb_4bit_compute_dtype=self.torch_dtype,
            )

        try:
            pipe = pipeline(
                "image-to-text",
                model=model_id,
                # "device_map" pode ser usado para autoalocar GPU se desejado.
                model_kwargs={"quantization_config": quantization_config}
                if quantization_config
                else None,
            )
            self.models[model_id] = pipe
            logging.info(f"Modelo '{model_id}' carregado com sucesso.")
            return pipe
        except Exception as e:
            logging.error(f"Erro ao carregar o modelo '{model_id}': {e}")
            raise RuntimeError(f"Falha ao carregar o modelo {model_id}. Erro: {str(e)}")

    def make_input_prompt(self, question: str) -> str:
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
        return f"USER: <image>\n{question}\nASSISTANT:"

    def process_output_text(self, output: str) -> str:
        """
        Remove partes desnecessárias do texto gerado, caso existam tags ou formatações
        que não queremos no resultado final.

        Parâmetros:
        -----------
        output : str
            Texto gerado pelo modelo.

        Retorna:
        --------
        str
            Texto processado, sem marcações extras.
        """
        pattern = r"USER:.*?ASSISTANT:\s*"
        return re.sub(pattern, "", output, flags=re.DOTALL).strip()

    def image_to_text(
        self,
        image_or_file: Union[str, bytes],
        question: str,
        model_id: Optional[str] = None,
        max_new_tokens: int = 1000,
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

        # Verifica se é string (caminho) ou bytes (arquivo binário)
        if isinstance(image_or_file, str):
            if not os.path.exists(image_or_file):
                raise FileNotFoundError(
                    f"Arquivo de imagem não encontrado: {image_or_file}"
                )
            image = Image.open(image_or_file)
        else:
            # Supondo que seja bytes
            image = Image.open(io.BytesIO(image_or_file))

        # Verifica dimensões
        if image.size[0] == 0 or image.size[1] == 0:
            raise ValueError("A imagem fornecida parece inválida ou vazia.")

        # Define o modelo a ser usado
        chosen_model_id = model_id or self.default_model_id
        pipe = self.load_model(chosen_model_id)

        # Monta o prompt
        prompt = self.make_input_prompt(question)
        logging.info(f"Realizando inferência com o modelo '{chosen_model_id}'...")

        # Realiza a inferência com a pipeline
        try:
            result = pipe(image, prompt=prompt, max_new_tokens=max_new_tokens)
            generated_text = result[0].get("generated_text", "")
            processed_output = self.process_output_text(generated_text)
            return processed_output
        except Exception as e:
            logging.error(f"Erro durante inferência: {e}")
            raise RuntimeError("Falha na geração de texto a partir da imagem.") from e
