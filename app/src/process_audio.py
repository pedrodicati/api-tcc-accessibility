import torch
import os
import logging
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from typing import Union, Dict


class AudioProcess:
    def __init__(self, model_id: str = ""):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.processor = AutoProcessor.from_pretrained(model_id, language="en")

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            use_safetensors=True,
        )
        self.model.to(device)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )
        logging.info("Model Speech Recognitivo loaded successfully.")

    def transcribe(self, audio_or_file: Union[str, bytes]) -> Dict[str, str]:
        if not isinstance(audio_or_file, str) and not isinstance(audio_or_file, bytes):
            raise ValueError("Audio must be a file path or bytes")
        
        if isinstance(audio_or_file, str):
            if not os.path.exists(audio_or_file):
                raise FileNotFoundError(f"File not found: {audio_or_file}")
            
            audio_or_file = open(audio_or_file, "rb").read()

        return self.pipe(audio_or_file)