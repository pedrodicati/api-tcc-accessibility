"""
Audio processing module for speech-to-text functionality.

This module provides functionality to transcribe audio content using
Hugging Face transformers and PyTorch, supporting multiple speech recognition models.
"""

import logging
import os
from typing import Dict, Optional, Union

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class AudioProcess:
    """
    Audio processing class for speech-to-text transcription.
    
    This class manages speech recognition models and provides functionality
    to transcribe audio content using Hugging Face transformers.
    
    Attributes:
    -----------
    device : torch.device
        The device (CPU or CUDA) for running inference.
    torch_dtype : torch.dtype
        The data type for tensors.
    default_model_id : str
        The default model identifier for speech recognition.
    models : Dict[str, pipeline]
        Dictionary storing loaded model pipelines.
    """
    
    def __init__(self, default_model_id: str = "openai/whisper-small") -> None:
        """
        Initialize the AudioProcess with a default speech recognition model.
        
        Parameters:
        -----------
        default_model_id : str
            The default model identifier to use for speech recognition.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.default_model_id = default_model_id
        self.models: Dict[str, pipeline] = {}

        logging.info(
            "AudioProcess iniciado. Device: %s, torch_dtype: %s", 
            self.device, self.torch_dtype
        )
        logging.info("Modelo padrão configurado para: %s", self.default_model_id)

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
                "Modelo '%s' já está carregado. Retornando instância existente.",
                model_id
            )
            return self.models[model_id]

        # Se não estiver carregado, criar uma nova pipeline
        logging.info("Carregando novo modelo: %s", model_id)

        processor = AutoProcessor.from_pretrained(model_id)

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            use_safetensors=True,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
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
        except Exception as exc:
            logging.error("Error loading model '%s': %s", model_id, str(exc))
            raise RuntimeError(f"Failed to load model {model_id}. Error: {str(exc)}") from exc

    def check_audio(self, audio_or_file: Union[str, bytes]) -> bytes:
        """
        Verifica se o áudio é uma string ou um arquivo e retorna o conteúdo do áudio.

        Parâmetros:
        -----------
        audio_or_file : Union[str, bytes]
            Áudio a ser transcrita.

        Retorna:
        --------
        bytes
            Conteúdo do áudio em bytes.
        """

        if not isinstance(audio_or_file, (str, bytes)):
            raise ValueError("Audio must be a file path or bytes")

        if isinstance(audio_or_file, str):
            if not os.path.exists(audio_or_file):
                raise FileNotFoundError(f"File not found: {audio_or_file}")

            with open(audio_or_file, "rb") as audio_file:
                audio_or_file = audio_file.read()

        return audio_or_file

    def transcribe(
        self, audio_or_file: Union[str, bytes], model_id: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Transcribe audio content to text using speech recognition models.
        
        Parameters:
        -----------
        audio_or_file : Union[str, bytes]
            Audio file path or audio content as bytes.
        model_id : Optional[str]
            Model ID to use for transcription. Uses default if None.
            
        Returns:
        --------
        Dict[str, str]
            Dictionary containing the transcribed text.
            
        Raises:
        -------
        RuntimeError
            If transcription fails.
        """
        audio = self.check_audio(audio_or_file)

        # Define o modelo a ser usado
        chosen_model_id = model_id or self.default_model_id
        pipe = self.load_model(chosen_model_id)

        try:
            return pipe(audio)
        except Exception as exc:
            logging.error("Error transcribing audio: %s", str(exc))
            raise RuntimeError(f"Failed to transcribe audio. Error: {str(exc)}") from exc
