from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, Union
import torch # For type hints related to torch.dtype if used directly

class Settings(BaseSettings):
    # Application settings
    LOG_LEVEL: str = "INFO"
    HF_TOKEN: Optional[str] = None

    # Default model configurations
    DEFAULT_AUDIO_MODEL_ID: str = "openai/whisper-small"
    DEFAULT_IMAGE_MODEL_ID: str = "llava-hf/llava-v1.6-mistral-7b-hf"

    # Device and DType settings
    # These can be "auto", or specific like "cuda", "cpu" for device
    # and "auto", "bfloat16", "float16", "float32" for dtype_str
    DEFAULT_DEVICE: str = "auto"
    DEFAULT_TORCH_DTYPE_STR: str = "auto" # Store as string, convert to torch.dtype in code

    # Image model specific settings (primarily for Hugging Face models)
    IMAGE_MODEL_MAX_NEW_TOKENS: int = 512

    # Quantization settings for Image Models (applied if device is CUDA)
    IMAGE_MODEL_QUANTIZATION_LOAD_IN_4BIT: bool = True
    IMAGE_MODEL_QUANTIZATION_BNB_4BIT_QUANT_TYPE: str = "nf4"
    IMAGE_MODEL_QUANTIZATION_BNB_4BIT_COMPUTE_DTYPE_STR: str = "bfloat16" # e.g., "bfloat16", "float16", "float32"
    IMAGE_MODEL_QUANTIZATION_BNB_4BIT_USE_DOUBLE_QUANT: bool = True
    IMAGE_MODEL_LOW_CPU_MEM_USAGE: bool = True # For from_pretrained

    # Ollama fallback settings
    OLLAMA_ENABLED: bool = True
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL_FOR_FALLBACK: str = "llava"
    OLLAMA_LLAMA3_2_VISION_MODEL: str = "llama3.2-vision" # Specific model for Llama 3.2 vision via Ollama

    # Audio model specific settings (can be expanded later)
    # AUDIO_MODEL_CHUNK_LENGTH_S: int = 30
    # AUDIO_MODEL_BATCH_SIZE: int = 16

    # File saving settings for analysis endpoint
    SAVE_ANALYSIS_FILES: bool = False
    ANALYSIS_FILES_SAVE_DIR: str = "processed_analysis_files"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        extra='ignore', # Ignore extra fields from .env
        # case_sensitive=False, # For environment variables if needed
    )

settings = Settings()

# It's generally cleaner if helper functions like get_torch_dtype are in a utils.py
# or directly within the modules that use them, rather than in settings.py.
# For this task, the conversion logic (string from settings -> torch.dtype object)
# will be implemented in process_audio.py and process_image.py.
