import torch
import os
import io
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from typing import List, Union


class ProcessAudio:
    def __init__(self, model_id: str = ""):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.processor = AutoProcessor.from_pretrained(model_id)

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
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

    def convert_bytes_to_numpy(self, audio_bytes: bytes) -> List[int]:

        return np.frombuffer(audio_bytes, np.int16).flatten().astype(np.float32) / 32768.0 

    def transcribe(self, audio_or_file: Union[str, bytes]):
        if isinstance(audio_or_file, str):
            if not os.path.exists(audio_or_file):
                raise FileNotFoundError(f"File not found: {audio_or_file}")
            
            return self.pipe(audio_or_file)
        elif isinstance(audio_or_file, bytes):
            audio = self.convert_bytes_to_numpy(audio_or_file)

            return self.pipe(audio)
