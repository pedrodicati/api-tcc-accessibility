import torch
import os
import logging
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from typing import Union, Dict, Optional


class AudioProcess:
    def __init__(self, default_model_id: str = "openai/whisper-small"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.default_model_id = default_model_id
        self.models: Dict[str, pipeline] = {}

        logging.info(
            f"AudioProcess iniciado. Device: {self.device}, torch_dtype: {self.torch_dtype}"
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

        processor = AutoProcessor.from_pretrained(model_id, language="en")

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            use_safetensors=True,
        )
        model.to(self.device)

        try:
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=self.torch_dtype,
                device=self.device,
            )
            self.models[model_id] = pipe

            logging.info("Model Speech Recognitivo loaded successfully.")

            return pipe
        except Exception as e:
            logging.error(f"Error loading model '{model_id}': {e}")
            raise RuntimeError(f"Failed to load model {model_id}. Error: {str(e)}")

    def transcribe(
        self, audio_or_file: Union[str, bytes], model_id: Optional[str] = None
    ) -> Dict[str, str]:
        if not isinstance(audio_or_file, str) and not isinstance(audio_or_file, bytes):
            raise ValueError("Audio must be a file path or bytes")

        if isinstance(audio_or_file, str):
            if not os.path.exists(audio_or_file):
                raise FileNotFoundError(f"File not found: {audio_or_file}")

            audio_or_file = open(audio_or_file, "rb").read()

        # Define o modelo a ser usado
        chosen_model_id = model_id or self.default_model_id
        pipe = self.load_model(chosen_model_id)

        return pipe(audio_or_file)
