import torch
from app.src.settings import settings # Import settings
import os
import logging
import gc # For garbage collection
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline as hf_pipeline
from typing import Union, Dict, Optional, Tuple, Any
from app.src.exceptions import ( # Import custom exceptions
    ModelNotFoundError, # Though less likely for ASR if default is always whisper
    ModelLoadError,
    InferenceError,
    InvalidInputError,
    FileProcessingError
)

logging.basicConfig(level=settings.LOG_LEVEL.upper())


# Helper function to convert string dtype from settings to torch.dtype
def str_to_torch_dtype(dtype_str: str) -> Optional[torch.dtype]:
    dtype_str = dtype_str.lower()
    if dtype_str == "bfloat16":
        return torch.bfloat16
    elif dtype_str == "float16" or dtype_str == "half":
        return torch.float16
    elif dtype_str == "float32" or dtype_str == "float":
        return torch.float32
    elif dtype_str == "auto":
        return None # Sentinel for auto-detection
    logging.warning(f"Unsupported torch_dtype string: {dtype_str}. Returning None.")
    return None

# Utility function for memory management
def _clear_gpu_memory():
    """Clears GPU memory and runs garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()
    logging.debug("GPU memory cleared and garbage collected.")

class AudioProcess:
    def __init__(
        self,
        default_model_id: Optional[str] = None,
        default_device: Optional[str] = None, # e.g. "cuda", "cpu", "auto"
        default_torch_dtype_str: Optional[str] = None # e.g. "bfloat16", "float16", "auto"
    ):
        self.default_model_id = default_model_id or settings.DEFAULT_AUDIO_MODEL_ID
        self.models: Dict[str, Tuple[hf_pipeline, Any, Any]] = {}
        self.current_model_id: Optional[str] = None
        self.current_pipeline: Optional[hf_pipeline] = None

        self.default_device_str = default_device or settings.DEFAULT_DEVICE
        self.default_torch_dtype_str = default_torch_dtype_str or settings.DEFAULT_TORCH_DTYPE_STR

        log_level = settings.LOG_LEVEL.upper()
        logging.getLogger().setLevel(log_level)
        logging.info(
            f"AudioProcess initialized. Default model: {self.default_model_id}, Device: {self.default_device_str}, DType: {self.default_torch_dtype_str}"
        )

    def _determine_device_and_dtype(
        self, device_req_str: Optional[str], dtype_req_str: Optional[str]
    ) -> Tuple[torch.device, torch.dtype]:
        """Determines the device and torch_dtype to use based on request or defaults from settings."""
        resolved_device_str = device_req_str or self.default_device_str
        if resolved_device_str.lower() == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(resolved_device_str.lower())

        resolved_dtype_str = dtype_req_str or self.default_torch_dtype_str
        torch_dtype_obj = str_to_torch_dtype(resolved_dtype_str)

        if torch_dtype_obj is None: # "auto" or invalid resolved to None
            if device.type == "cuda":
                if torch.cuda.is_bf16_supported():
                    torch_dtype_obj = torch.bfloat16
                    logging.info("Auto-selected bfloat16 for CUDA device.")
                else:
                    torch_dtype_obj = torch.float16
                    logging.info("Auto-selected float16 for CUDA device (bfloat16 not supported).")
            else: # CPU
                torch_dtype_obj = torch.float32
                logging.info("Auto-selected float32 for CPU device.")

        logging.info(f"Resolved device: {device}, Resolved torch_dtype: {torch_dtype_obj}")
        return device, torch_dtype_obj

    def _load_model_resources(
        self,
        model_id: str,
        device: torch.device, # Expects resolved torch.device
        torch_dtype: torch.dtype # Expects resolved torch.dtype
    ) -> Tuple[hf_pipeline, Any, Any]:
        """
        Loads the ASR model, processor, and creates the pipeline.
        """
        logging.info(f"Attempting to load ASR model resources for: {model_id} on {device} with {torch_dtype}")
        _clear_gpu_memory()

        load_kwargs = {
            "token": settings.HF_TOKEN if settings.HF_TOKEN else None,
        }
        if not load_kwargs["token"]:
            del load_kwargs["token"]

        try:
            processor = AutoProcessor.from_pretrained(model_id, **load_kwargs)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                use_safetensors=True,
                torch_dtype=torch_dtype if device.type == "cuda" else torch.float32,
                low_cpu_mem_usage=settings.IMAGE_MODEL_LOW_CPU_MEM_USAGE, # Using this global setting
                **load_kwargs
            )

            if not getattr(model, 'hf_device_map', None):
                model.to(device)

            pipe = hf_pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=torch_dtype,
                device=device if device.type != 'mps' else None,
            )
            self.models[model_id] = (pipe, model, processor)
            logging.info(f"ASR model '{model_id}' loaded successfully on {device}.")
            return pipe, model, processor
        except Exception as e: # Catching a broad exception here from HuggingFace or network
            logging.error(f"Error loading ASR model '{model_id}': {e}", exc_info=True)
            # Could be unknown model, network issue, config error. ModelLoadError is appropriate.
            raise ModelLoadError(f"Failed to load ASR model '{model_id}'. Original error: {str(e)}") from e

    def _get_or_load_pipeline(
        self,
        model_id: str,
        device_req_str: Optional[str] = None, # String from user/endpoint
        dtype_req_str: Optional[str] = None  # String from user/endpoint
    ) -> hf_pipeline:
        """
        Retrieves an ASR pipeline from cache or loads it.
        """
        pipe_tuple = self.models.get(model_id)
        # Resolve target device and dtype based on request or defaults
        target_device, target_dtype = self._determine_device_and_dtype(device_req_str, dtype_req_str)

        if pipe_tuple is None:
            logging.info(f"ASR pipeline for '{model_id}' not in cache. Loading on {target_device} with {target_dtype}...")
            pipe, _, _ = self._load_model_resources(model_id, target_device, target_dtype)
        else:
            pipe, loaded_model, processor = pipe_tuple
            current_pipeline_device = pipe.device
            current_model_dtype = loaded_model.dtype # Assuming model has dtype attribute

            # If device or dtype mismatch, and not using device_map, consider re-configuration
            if (current_pipeline_device != target_device or current_model_dtype != target_dtype) and \
               not getattr(loaded_model, 'hf_device_map', None):
                logging.info(f"ASR model {model_id} is on {current_pipeline_device}/{current_model_dtype}, but {target_device}/{target_dtype} was requested. Re-configuring pipeline.")
                _clear_gpu_memory()

                if hasattr(loaded_model, 'to'): loaded_model.to(target_device, dtype=target_dtype)

                pipe = hf_pipeline(
                    "automatic-speech-recognition", model=loaded_model, tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor, torch_dtype=target_dtype,
                    device=target_device if target_device.type != 'mps' else None,
                )
                self.models[model_id] = (pipe, loaded_model, processor)
                logging.info(f"ASR pipeline for {model_id} re-configured for device {target_device} and dtype {target_dtype}.")
            else:
                logging.info(f"ASR pipeline for '{model_id}' found in cache and on compatible device/dtype ({current_pipeline_device}/{current_model_dtype}).")

        self.current_model_id = model_id
        self.current_pipeline = pipe
        return self.current_pipeline

    def set_model(
        self,
        model_id: str,
        load_now: bool = False,
        device_str: Optional[str] = None, # String for API
        torch_dtype_str: Optional[str] = None # String for API
    ) -> None:
        """
        Sets the ASR model to be used for subsequent calls. Can optionally pre-load.
        """
        target_device, target_dtype = self._determine_device_and_dtype(device_str, torch_dtype_str)

        target_device, target_dtype = self._determine_device_and_dtype(device_str, torch_dtype_str)

        # Check if the requested model is already the current one and loaded with compatible settings.
        current_pipeline_tuple = self.models.get(self.current_model_id if self.current_model_id else model_id)
        if self.current_model_id == model_id and self.current_pipeline is not None and current_pipeline_tuple:
            # If it's already the current model, check if device/dtype match, or if no specific device/dtype requested for change
            if (device_str is None or current_pipeline_tuple[0].device == target_device) and \
               (torch_dtype_str is None or current_pipeline_tuple[1].dtype == target_dtype):
                msg = f"ASR model '{model_id}' is already active and loaded with compatible settings."
                logging.info(msg)
                # If load_now is True, we still ensure it's loaded, which _get_or_load_pipeline will do.
                # If not load_now, and it's current, we can return.
                if not load_now:
                    return {"message": msg}
            else:
                logging.info(f"ASR model '{model_id}' is current, but new device/dtype requested. Will reload if load_now=True or on next use.")

        logging.info(f"Setting ASR model from '{self.current_model_id}' to '{model_id}'. Target device: {target_device}, Target Dtype: {target_dtype}. Load now: {load_now}")

        if self.current_model_id and self.current_model_id != model_id and self.models.get(self.current_model_id):
            old_pipe_tuple = self.models.get(self.current_model_id)
            if old_pipe_tuple and old_pipe_tuple[0].device.type == 'cuda':
                logging.info(f"Clearing GPU memory for old ASR model: {self.current_model_id}")
                if self.current_model_id in self.models:
                    del self.models[self.current_model_id] # Remove from cache
                self.current_pipeline = None # Explicitly release
                _clear_gpu_memory()

        self.default_model_id = model_id # Update the instance's default model ID
        self.current_model_id = model_id # Set current model ID
        self.current_pipeline = None    # Ensure it's reloaded if different or params change

        if load_now:
            try:
                self._get_or_load_pipeline(model_id, device_str, torch_dtype_str)
                msg = f"ASR model '{model_id}' set and loaded successfully on {target_device} with {target_dtype}."
                logging.info(msg)
                return {"message": msg}
            except ModelLoadError as e: # Catch specific load error
                logging.error(f"Failed to pre-load ASR model '{model_id}': {e}", exc_info=True)
                raise # Re-raise the original ModelLoadError
            except Exception as e: # Catch any other unexpected errors during load
                logging.error(f"Unexpected error during pre-load of ASR model '{model_id}': {e}", exc_info=True)
                raise ModelLoadError(f"Unexpected error pre-loading ASR model '{model_id}': {str(e)}") from e
        else:
            msg = f"ASR model set to '{model_id}'. It will be loaded on next use with target device {target_device} and dtype {target_dtype}."
            logging.info(msg)
            return {"message": msg}


    def check_audio(self, audio_or_file: Union[str, bytes]) -> bytes:
        """
        Verifies audio input and returns audio bytes.
        Raises FileProcessingError for issues.
        """
        try:
            if isinstance(audio_or_file, str):
                if not os.path.exists(audio_or_file):
                    raise FileProcessingError(f"Audio file not found at path: {audio_or_file}")
                with open(audio_or_file, "rb") as f:
                    audio_bytes = f.read()
                return audio_bytes
            elif isinstance(audio_or_file, bytes):
                if not audio_or_file:
                    raise FileProcessingError("Audio bytes input is empty.")
                return audio_or_file
            else:
                # This case should ideally be caught by FastAPI's type validation if type hints are precise for endpoint.
                # If it reaches here, it's an unexpected type.
                raise InvalidInputError(f"Audio input must be a file path (str) or bytes, got {type(audio_or_file)}.")
        except FileNotFoundError as e: # Should be caught by os.path.exists, but as a safeguard
             raise FileProcessingError(f"Audio file not found: {e}")
        except Exception as e: # Catch other potential OS or read errors
            raise FileProcessingError(f"Error processing audio file: {e}")


    def transcribe(
        self,
        audio_or_file: Union[str, bytes],
        model_id: Optional[str] = None,
        device_str: Optional[str] = None,
        torch_dtype_str: Optional[str] = None, # String for dtype
        chunk_length_s: Optional[int] = 30,
        batch_size: Optional[int] = None,
        return_timestamps: Union[bool, str] = True,
    ) -> Dict[str, Any]:
        """
        Transcribes audio using the specified or default ASR model.
        """
        try:
            audio_bytes = self.check_audio(audio_or_file)

            target_model_id = model_id or self.current_model_id or self.default_model_id
            if not target_model_id:
                 target_model_id = settings.DEFAULT_AUDIO_MODEL_ID
                 logging.warning(f"No model ID specified for transcribe, using settings default: {target_model_id}")

            active_pipeline = self._get_or_load_pipeline(target_model_id, device_str, torch_dtype_str)

            pipeline_kwargs = {"return_timestamps": return_timestamps}
            if chunk_length_s is not None: pipeline_kwargs["chunk_length_s"] = chunk_length_s
            if batch_size is not None: pipeline_kwargs["batch_size"] = batch_size

            actual_pipeline_device = active_pipeline.device
            logging.info(f"Transcribing audio with ASR model '{target_model_id}' on device {actual_pipeline_device} with args: {pipeline_kwargs}")

            result = active_pipeline(audio_bytes, **pipeline_kwargs)
            if result is None or not isinstance(result, dict) or "text" not in result:
                 logging.warning(f"ASR pipeline for model '{target_model_id}' returned an unexpected result: {result}")
                 raise InferenceError(f"ASR model '{target_model_id}' produced an empty or invalid result.")
            return result
        except (ModelLoadError, FileProcessingError, InvalidInputError) as e: # Propagate known errors
            raise
        except Exception as e: # Catch-all for other errors during transcription
            logging.error(f"Error transcribing audio with model '{target_model_id}': {e}", exc_info=True)
            raise InferenceError(f"Failed to transcribe audio with model '{target_model_id}'. Error: {str(e)}") from e
