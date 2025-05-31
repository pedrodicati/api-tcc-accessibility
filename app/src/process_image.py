import torch
from app.src.settings import settings # Import settings
import logging
import re
import io
import os
import gc
import warnings
import traceback

from PIL import Image
from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    # MllamaForConditionalGeneration, # This model seems to be causing issues, will handle fallback
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info
from .ollama_process import OllamaProcess
from app.src.exceptions import ( # Import custom exceptions
    ModelNotFoundError,
    ModelLoadError,
    InferenceError,
    InvalidInputError,
    OllamaNotAvailableError,
    FileProcessingError
)
from typing import Union, List, Dict, Optional, Tuple, Any

# Configure logging based on settings before it's used by other modules if possible
# However, basicConfig should ideally be called once.
# We will assume LOG_LEVEL from settings is handled by a central logging config if needed,
# or here if this is the main entry point for this class's logging.
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
    elif dtype_str == "auto": # Let system decide based on hardware / other params
        return None # Sentinel for auto-detection in _determine_device_and_dtype
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


class ImageProcess:
    def __init__(
        self,
        # Defaults will now primarily come from settings object
        default_model_id: Optional[str] = None,
        ollama_instance: Optional[OllamaProcess] = None,
        default_device: Optional[str] = None, # e.g., "cuda", "cpu", "auto"
        default_torch_dtype_str: Optional[str] = None, # e.g, "bfloat16", "float16", "auto"
        use_ollama_fallback_for: Optional[List[str]] = None,
        force_ollama_for_all: Optional[bool] = None
    ) -> None:
        self.default_model_id = default_model_id or settings.DEFAULT_IMAGE_MODEL_ID
        self.ollama_instance = ollama_instance # Must be provided if Ollama is to be used

        self.models: Dict[str, Tuple[Any, Any, str]] = {}
        self.current_model_id: Optional[str] = None
        self.current_model: Optional[Any] = None
        self.current_processor: Optional[Any] = None

        self.default_device_str = default_device or settings.DEFAULT_DEVICE
        self.default_torch_dtype_str = default_torch_dtype_str or settings.DEFAULT_TORCH_DTYPE_STR

        # Fallback configuration
        self.use_ollama_fallback_for = use_ollama_fallback_for or ["meta-llama/Llama-3.2-11B-Vision-Instruct"] # Default list
        self.force_ollama_for_all = force_ollama_for_all if force_ollama_for_all is not None else False
        if not settings.OLLAMA_ENABLED: # Global override from settings
            self.ollama_instance = None
            self.force_ollama_for_all = False
            logging.info("Ollama is globally disabled via settings. Fallback will not be available.")


        log_level = settings.LOG_LEVEL.upper()
        logging.getLogger().setLevel(log_level) # Ensure logger level is set
        logging.info(
            f"ImageProcess initialized. Default model: {self.default_model_id}, Device: {self.default_device_str}, DType: {self.default_torch_dtype_str}"
        )

    def _determine_device_and_dtype(
        self, device_req_str: Optional[str], dtype_req_str: Optional[str] # Now expects string for dtype
    ) -> Tuple[torch.device, torch.dtype]:
        """Determines the device and torch_dtype to use based on request or defaults from settings."""

        # Determine device
        resolved_device_str = device_req_str or self.default_device_str
        if resolved_device_str.lower() == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(resolved_device_str.lower())

        # Determine dtype
        resolved_dtype_str = dtype_req_str or self.default_torch_dtype_str
        torch_dtype_obj = str_to_torch_dtype(resolved_dtype_str)

        if torch_dtype_obj is None: # "auto" or invalid string was converted to None
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
        model_id_to_load: str,
        # These are now the resolved torch.device and torch.dtype objects
        device: torch.device,
        torch_dtype: torch.dtype,
    ) -> Tuple[Any, Any]:
        """
        Loads the model and processor for the given model_id.
        Uses device and torch_dtype objects passed directly.
        Returns tuple (model, processor)
        """
        logging.info(f"Attempting to load model resources for: {model_id_to_load} on {device} with {torch_dtype}")
        _clear_gpu_memory()

        model = None
        processor = None

        # quantization_config is now built using settings
        quantization_config = None
        bnb_compute_dtype = str_to_torch_dtype(settings.IMAGE_MODEL_QUANTIZATION_BNB_4BIT_COMPUTE_DTYPE_STR)
        if bnb_compute_dtype is None: # Handle auto or default for compute_dtype if not properly set
            bnb_compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            logging.warning(f"BNB compute dtype auto-set to {bnb_compute_dtype} due to invalid/auto setting '{settings.IMAGE_MODEL_QUANTIZATION_BNB_4BIT_COMPUTE_DTYPE_STR}'")


        if device.type == "cuda" and settings.IMAGE_MODEL_QUANTIZATION_LOAD_IN_4BIT:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=settings.IMAGE_MODEL_QUANTIZATION_LOAD_IN_4BIT,
                bnb_4bit_quant_type=settings.IMAGE_MODEL_QUANTIZATION_BNB_4BIT_QUANT_TYPE,
                bnb_4bit_compute_dtype=bnb_compute_dtype,
                bnb_4bit_use_double_quant=settings.IMAGE_MODEL_QUANTIZATION_BNB_4BIT_USE_DOUBLE_QUANT,
            )
            logging.info(f"Using BitsAndBytes quantization: {quantization_config.to_dict()}")

        hf_model_load_kwargs = {
            "torch_dtype": torch_dtype if device.type == "cuda" else torch.float32, # float32 for CPU always
            "low_cpu_mem_usage": settings.IMAGE_MODEL_LOW_CPU_MEM_USAGE,
            "quantization_config": quantization_config if device.type == "cuda" else None,
            "device_map": "auto", # Let transformers handle multi-GPU or auto placement
            "token": settings.HF_TOKEN if settings.HF_TOKEN else None,
        }
        # Remove None token from kwargs
        if not hf_model_load_kwargs["token"]:
            del hf_model_load_kwargs["token"]

        try:
            if model_id_to_load == "llava-hf/llava-v1.6-mistral-7b-hf":
                processor = LlavaNextProcessor.from_pretrained(model_id_to_load, token=settings.HF_TOKEN if settings.HF_TOKEN else None)
                model = LlavaNextForConditionalGeneration.from_pretrained(model_id_to_load, **hf_model_load_kwargs)
            elif model_id_to_load == "Qwen/Qwen2.5-VL-7B-Instruct":
                processor = AutoProcessor.from_pretrained(model_id_to_load, token=settings.HF_TOKEN if settings.HF_TOKEN else None)
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id_to_load, **hf_model_load_kwargs)
            elif model_id_to_load == "meta-llama/Llama-3.2-11B-Vision-Instruct":
                logging.warning(f"Attempting to load large model: {model_id_to_load}. This might fail and fallback to Ollama if configured.")
                from transformers import MllamaForConditionalGeneration
                processor = AutoProcessor.from_pretrained(model_id_to_load, token=settings.HF_TOKEN if settings.HF_TOKEN else None)
                model = MllamaForConditionalGeneration.from_pretrained(model_id_to_load, **hf_model_load_kwargs)
            else:
                warnings.warn(f"ImageProcess attempting generic loading for '{model_id_to_load}'.", UserWarning)
                processor = AutoProcessor.from_pretrained(model_id_to_load, use_fast=True, token=settings.HF_TOKEN if settings.HF_TOKEN else None)
                model = AutoModelForImageTextToText.from_pretrained(model_id_to_load, **hf_model_load_kwargs)

            # device_map="auto" handles model.to(device), so explicit .to(device) is usually not needed
            # if model and hasattr(model, 'to') and not getattr(model, 'hf_device_map', None):
            #      model.to(device) # Ensure model is on the final resolved device if not device_mapped

            logging.info(f"Successfully loaded Hugging Face model: {model_id_to_load} using device_map='auto'. Effective model device: {model.device if hasattr(model, 'device') else 'unknown'}")
            self.models[model_id_to_load] = (model, processor, "hf")
            return model, processor

        except Exception as e:
            logging.error(f"Error loading Hugging Face model '{model_id_to_load}': {e}\n{traceback.format_exc()}")
            if self.ollama_instance and settings.OLLAMA_ENABLED and \
               (self.force_ollama_for_all or model_id_to_load in self.use_ollama_fallback_for):
                logging.info(f"Falling back to Ollama for model: {model_id_to_load}")

                ollama_model_to_use = settings.OLLAMA_MODEL_FOR_FALLBACK # Default fallback
                if model_id_to_load == "meta-llama/Llama-3.2-11B-Vision-Instruct":
                    ollama_model_to_use = settings.OLLAMA_LLAMA3_2_VISION_MODEL
                elif "llava" in model_id_to_load.lower(): # if it's some other llava variant
                    ollama_model_to_use = "llava" # or settings.OLLAMA_MODEL_FOR_FALLBACK if more generic
                # Add more specific mappings if needed based on model_id_to_load

                logging.info(f"Attempting to use Ollama model: {ollama_model_to_use}")
                try:
                    if hasattr(self.ollama_instance, 'set_model'):
                        self.ollama_instance.set_model(ollama_model_to_use)
                    elif hasattr(self.ollama_instance, 'model') and self.ollama_instance.model != ollama_model_to_use:
                        logging.warning(f"Ollama instance model ({self.ollama_instance.model}) differs from target fallback ({ollama_model_to_use}). Re-initialize OllamaProcess or ensure set_model is effective if this is an issue.")

                    model_ollama = self.ollama_instance
                    processor_ollama = None
                    logging.info(f"Successfully configured Ollama fallback for: {model_id_to_load} with Ollama model: {ollama_model_to_use}")
                    self.models[model_id_to_load] = (model_ollama, processor_ollama, "ollama")
                    return model_ollama, processor_ollama
                except Exception as ollama_e: # Catch specific errors from Ollama if possible
                    logging.error(f"Ollama fallback attempt failed for {model_id_to_load} (Ollama model {ollama_model_to_use}): {ollama_e}", exc_info=True)
                    # Re-raise as ModelLoadError, indicating the fallback also failed.
                    raise ModelLoadError(f"HuggingFace model '{model_id_to_load}' failed to load (Error: {e}). Ollama fallback also failed (Error: {ollama_e}).") from ollama_e

            # If Ollama is not enabled or instance not available, or not in fallback list
            raise ModelLoadError(f"Failed to load HuggingFace model '{model_id_to_load}'. Error: {str(e)}") from e

    def _get_or_load_model(
        self,
        model_id_to_use: str,
        device_req_str: Optional[str] = None,
        dtype_req_str: Optional[str] = None # Now string
    ) -> Tuple[Any, Any, str]:
        if model_id_to_use not in self.models:
            logging.info(f"Model '{model_id_to_use}' not in cache. Loading...")
            # Determine device and dtype for loading using _determine_device_and_dtype
            # This now returns torch.device and torch.dtype objects
            resolved_device, resolved_dtype = self._determine_device_and_dtype(device_req_str, dtype_req_str)
            self._load_model_resources(model_id_to_use, resolved_device, resolved_dtype)
        else:
            logging.info(f"Model '{model_id_to_use}' found in cache.")

        model, processor, model_type_hint = self.models[model_id_to_use]

        self.current_model_id = model_id_to_use
        self.current_model = model
        self.current_processor = processor

        return self.current_model, self.current_processor, model_type_hint

    def set_model(
        self,
        model_id: str,
        load_now: bool = False,
        device_str: Optional[str] = None, # Keep as string for API consistency
        torch_dtype_str: Optional[str] = None # Keep as string
    ) -> None:
        # Check if model is already current and loaded with compatible settings (more complex check needed if device/dtype can change for existing loaded model)
        # For now, if model_id is the same and it's in cache, assume it's fine.
        # A more robust check would verify if the cached model's device/dtype match new device_str/torch_dtype_str
        # More robust check for existing model and compatibility
        existing_model_data = self.models.get(model_id)
        if self.current_model_id == model_id and existing_model_data is not None:
            # TODO: Add check for device/dtype compatibility if device_str/torch_dtype_str are provided
            # For now, if it's the current model and cached, assume it's fine unless load_now forces re-evaluation
            if not load_now:
                 logging.info(f"Model '{model_id}' is already set and loaded. No changes made.")
                 return
            else:
                 logging.info(f"Model '{model_id}' is already set. 'load_now' is True, will ensure it's loaded with specified params (if any).")


        logging.info(f"Setting active model to: {model_id}. Load now: {load_now}. Device req: {device_str}, DType req: {torch_dtype_str}")

        if self.current_model_id and self.current_model_id != model_id and self.models.get(self.current_model_id):
            logging.info(f"Switching from {self.current_model_id} to {model_id}. Clearing GPU memory.")
            _clear_gpu_memory()

        self.current_model_id = model_id
        self.current_model = None
        self.current_processor = None

        if load_now:
            if self.current_model_id != model_id : _clear_gpu_memory()
            try:
                _, _, _ = self._get_or_load_model(model_id, device_str, torch_dtype_str)
                logging.info(f"Model {model_id} pre-loaded successfully using device_req: {device_str}, dtype_req: {torch_dtype_str}.")
            except ModelLoadError as mle: # Catch specific load error
                logging.error(f"Failed to pre-load model {model_id}: {mle}", exc_info=True)
                # self.current_model_id = settings.DEFAULT_IMAGE_MODEL_ID
                raise # Re-raise the original ModelLoadError
            except Exception as e: # Catch any other unexpected errors during load
                logging.error(f"Unexpected error during pre-load of model {model_id}: {e}", exc_info=True)
                # self.current_model_id = settings.DEFAULT_IMAGE_MODEL_ID
                raise ModelLoadError(f"Unexpected error pre-loading model {model_id}: {str(e)}") from e

    def make_input_prompt(
        self, question: str, image: Image.Image, model_id_for_template: str, processor_for_template: Optional[Any], model_type_hint: str
    ) -> Union[str, List[Dict[str, Any]]]: # This method should be fine as is, uses already loaded model/proc.
        system_prompt = (
            "Você é um assistente virtual projetado para ajudar pessoas com deficiência visual "
            "a compreenderem o ambiente ao seu redor de forma clara, objetiva e fiel. Sua principal tarefa "
            "é descrever com precisão o que está presente na imagem fornecida, destacando detalhes importantes "
            "que permitam ao usuário construir uma imagem mental do ambiente. Seja conciso e evite descrições excessivamente longas. "
            "Concentre-se em informações relevantes, como objetos, pessoas, expressões faciais, cores e posições. "
            "Somente mencione riscos ou perigos se eles estiverem visíveis na imagem, explicando de forma direta "
            "o que é o risco e, se necessário, o que o usuário pode fazer para evitá-lo. "
            "Se não houver riscos, foque exclusivamente na descrição fiel da cena. "
            "Mantenha sempre um tom educado, objetivo e prestativo."
        )

        prompt: Union[str, List[Dict[str, Any]]]

        if model_type_hint == "hf" and processor_for_template and hasattr(processor_for_template, 'apply_chat_template'):
            # Generic chat structure for multimodal models
            # Image is often a placeholder here; actual image data is passed to processor() call
            chat_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "text", "text": question}, {"type": "image"}]}
            ]
            # Some processors might not want the image in `apply_chat_template` but as a direct kwarg in `processor()`
            # For Llava, the prompt is usually just the text part, and image is passed to processor()
            if "llava" in model_id_for_template.lower():
                 # Llava's template is often "USER: <image>\n{question} ASSISTANT:"
                 # The processor handles adding <image> if `images` argument is present.
                 # So, the text for `apply_chat_template` should be just the user question.
                 # Or, if the system prompt is desired, it should be part of the user question string.
                 # Let's try with system prompt then user question.
                chat_messages_llava = [
                    {"role": "user", "content": f"{system_prompt} USER: {question} ASSISTANT:"} # Simpler, direct prompt
                ]
                # However, LlavaNextProcessor expects a list of messages for `apply_chat_template`
                # For `text="USER: <image>\n{question} ASSISTANT:"`, image is passed to `processor()`
                # The `apply_chat_template` for Llava might expect:
                # [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is this?"}]}]
                # Let's use the structure that includes the image placeholder for the template.
                # The processor call `processor(text=prompt, images=image)` will then use this.
                # The `prompt` variable here is the *text* part of the multimodal input.
                # The original code's `make_input_prompt` returned a string prompt for Llava.
                # e.g. self.processor.apply_chat_template(chat, add_generation_prompt=True)
                # where chat was: [{"role": "system", ...}, {"role": "user", ...}]
                # Let's make `prompt` the string that goes into `processor(text=prompt, ...)`

                # For Llava, the processor itself often formats the full prompt string including image tokens
                # when you pass `images` argument to it.
                # The `text` argument to the processor should be the textual part of the prompt.
                # `apply_chat_template` can generate this text.

                # Let's simplify: the prompt for Llava type models will be the question,
                # and the system prompt can be part of it if needed.
                # The processor will handle <image> token.
                if "llava-hf/llava-v1.6-mistral-7b-hf" in model_id_for_template:
                    # This is the structure used in Llava's docs for `processor()`
                    prompt = f"USER: {question}\nASSISTANT:"
                    # No, apply_chat_template is preferred if available
                    # chat_for_llava_template = [{"role": "user", "content": [{"type": "text", "text": question}]}] # Removed image from here
                    # prompt = processor_for_template.apply_chat_template(chat_for_llava_template, add_generation_prompt=True, tokenize=False)
                    # The original code for llava used:
                    # prompt from make_input_prompt (which used apply_chat_template on a chat with system and user messages)
                    # inputs = self.processor(text=prompt, images=image, return_tensors="pt")
                    # So, we need `prompt` to be the output of `apply_chat_template`.
                    chat_for_template = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": [{"type": "text", "text": question }]} # Image placeholder is implicit for Llava processor
                    ]
                    prompt = processor_for_template.apply_chat_template(chat_for_template, add_generation_prompt=True, tokenize=False)


                elif "Qwen" in model_id_for_template:
                    # Qwen's apply_chat_template is generally robust.
                    # The original code used `process_vision_info` later, which might be complex.
                    # If `apply_chat_template` can take multimodal content directly, that's better.
                    # Qwen expects content in a specific list format.
                    qwen_chat_content = [ {"type": "text", "text": system_prompt} ] # System first
                    qwen_chat_content.append( {"type": "image"} ) # Image placeholder
                    qwen_chat_content.append( {"type": "text", "text": question} )
                    chat_for_qwen_template = [{"role": "user", "content": qwen_chat_content}]
                    # The processor might take this complex structure.
                    # Original: processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                    # where chat was: [{"role": "system", ...}, {"role": "user", "content": [{"type":"image"}, {"type":"text"}]}]
                    final_chat_for_qwen = [
                        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
                    ]
                    prompt = processor_for_template.apply_chat_template(final_chat_for_qwen, tokenize=False, add_generation_prompt=True)
                else: # Generic HF model with apply_chat_template
                    prompt = processor_for_template.apply_chat_template(
                        chat_messages, tokenize=False, add_generation_prompt=True
                    )
            else: # Fallback if no apply_chat_template or not HF
                prompt = f"{system_prompt}\nUSER: {question}\nASSISTANT:"

        elif model_type_hint == "ollama" and self.ollama_instance:
            prompt = self.ollama_instance.make_prompt(prompt=question, image=image, system_prompt=system_prompt)
        else: # Should not happen if model loaded correctly
            logging.error(f"Cannot make prompt for model {model_id_for_template}, type {model_type_hint}, processor missing or incompatible.")
            prompt = f"{system_prompt}\nUSER: {question}\nASSISTANT:" # Basic fallback

        logging.debug(f"Generated prompt for {model_id_for_template} (type: {model_type_hint}): {prompt if isinstance(prompt, str) else 'List prompt'}")
        return prompt

    def process_output_text(self, output: str) -> str:
        output = re.sub(r"<<SYS>>.*?<</SYS>>", "", output, flags=re.DOTALL).strip()
        output = re.sub(r"system .*?assistant ", "", output, flags=re.DOTALL).strip()
        output = re.sub(r"\[INST\].*?\[/INST\]", "", output, flags=re.DOTALL).strip()
        output = re.sub(r"\s+([.,!?])", r"\1", output)
        output = output.replace("\n", " ").strip()
        # Further clean common model artifacts like "ASSISTANT:" or "GPT:" if they appear at start
        output = re.sub(r"^(ASSISTANT|GPT): ?", "", output).strip()
        return output

    def check_image(self, image_or_file: Union[str, bytes, Image.Image]) -> Image.Image:
        try:
            if isinstance(image_or_file, Image.Image):
                image = image_or_file
            elif isinstance(image_or_file, str):
                if not os.path.exists(image_or_file):
                    raise FileProcessingError(f"Image file not found at path: {image_or_file}")
                image = Image.open(image_or_file)
            elif isinstance(image_or_file, bytes):
                try:
                    image = Image.open(io.BytesIO(image_or_file))
                except Exception as e:
                    raise FileProcessingError(f"Could not open image from bytes: {e}")
            else:
                raise InvalidInputError(f"image_or_file must be a PIL Image, path (str), or bytes, got {type(image_or_file)}")

            if image.mode in ["RGBA", "P", "LA", "L"]: # Convert common modes to RGB
                original_mode = image.mode
                image = image.convert("RGB")
                logging.info(f"Converted image from {original_mode} to RGB.")

            if image.size[0] == 0 or image.size[1] == 0:
                raise FileProcessingError("The image provided is empty or has zero dimensions.")
            return image
        except FileNotFoundError as fnfe: # Re-raise as our custom type
            raise FileProcessingError(str(fnfe))
        except ValueError as ve: # Re-raise PILS's ValueError as our type
            raise FileProcessingError(f"Invalid image file: {ve}")


    def image_preprocess(self, image: Image.Image) -> Image.Image:
        return image

    def image_to_text(
        self,
        image_or_file: Union[str, bytes, Image.Image],
        question: str,
        model_id: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        device_req_str: Optional[str] = None,
        dtype_req_str: Optional[str] = None
    ) -> str:
        if not question or not question.strip():
            raise InvalidInputError("Question cannot be empty.")

        target_model_id = model_id or self.current_model_id or self.default_model_id
        if not target_model_id:
            target_model_id = settings.DEFAULT_IMAGE_MODEL_ID
            logging.warning(f"No model ID specified, using settings default: {target_model_id}")

        final_max_new_tokens = max_new_tokens if max_new_tokens is not None else settings.IMAGE_MODEL_MAX_NEW_TOKENS

        active_model = None
        model_type = "unknown"
        try:
            image = self.check_image(image_or_file)

            active_model, active_processor, model_type = self._get_or_load_model(
                target_model_id, device_req_str, dtype_req_str
            )

            effective_device = torch.device("cpu")
            if model_type == "hf" and active_model and hasattr(active_model, 'device'):
                effective_device = active_model.device
            elif model_type == "ollama" and self.ollama_instance and hasattr(self.ollama_instance, 'device'):
                 effective_device = self.ollama_instance.device
            else:
                 temp_device, _ = self._determine_device_and_dtype(device_req_str, dtype_req_str)
                 effective_device = temp_device

            logging.info(f"Preparing for inference with model '{target_model_id}' (type: {model_type}) on effective device {effective_device}")
            text_prompt_or_chat_payload = self.make_input_prompt(question, image, target_model_id, active_processor, model_type)

            generated_text = ""

            if model_type == "ollama":
                if not self.ollama_instance or not settings.OLLAMA_ENABLED:
                    raise OllamaNotAvailableError("Ollama is not enabled or configured for use.")
                if not isinstance(text_prompt_or_chat_payload, (list, dict, str)):
                     raise InvalidInputError(f"Prompt for Ollama is not in the expected format, got {type(text_prompt_or_chat_payload)}")
                generated_text = active_model.predict_model(text_prompt_or_chat_payload, max_new_tokens=final_max_new_tokens)
            elif model_type == "hf" and active_processor and active_model:
                inputs = None
                if "Qwen" in target_model_id:
                    # For Qwen, `text_prompt_or_chat_payload` is the string from `apply_chat_template`.
                    # `process_vision_info` is used to prepare image/video inputs.
                    # The original `chat` for `process_vision_info` was like:
                    # [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
                    # The `image` object needs to be embedded here.
                    qwen_chat_for_vision_processing = [
                        {"role": "user", "content": [{"type": "image", "_pil_image": image}, {"type": "text", "text": question}]}
                    ]
                    # Assuming qwen_vl_utils.process_vision_info can find the PIL image if passed this way.
                    # Or it might expect file paths or URLs. Adjust if needed.
                    # If process_vision_info needs the raw image, pass `image` directly.
                    # The function signature of process_vision_info is (messages, processor, **kwargs)
                    # It extracts image/video based on "image" or "video" keys in content dicts.
                    # So, let's ensure the image is there.

                    # The `text_prompt_or_chat_payload` is already the fully formatted prompt string from apply_chat_template.
                    # We need to prepare the image inputs for the `images` kwarg of the processor.
                    # `process_vision_info` is supposed to return these.

                    # Let's align with how Qwen's own examples usually work:
                    # 1. Prepare `messages` for `apply_chat_template` (done in `make_input_prompt`) -> `text_prompt_or_chat_payload`
                    # 2. Separately process the image with the processor if needed, or pass it directly.
                    # The original code did:
                    #   image_inputs, video_inputs = process_vision_info(chat) -> chat here was the structured one.
                    #   inputs = self.processor(text=prompt, images=image_inputs, videos=video_inputs)
                    # This implies `process_vision_info` is a preprocessing step for the `images` arg.

                    # Let's simplify the Qwen data path if possible, or replicate original if necessary.
                    # For now, assume `active_processor` can handle `images=image` directly if `text` is from `apply_chat_template`.
                    # If `process_vision_info` is essential, it needs to be called here correctly.
                    # The `text_prompt_or_chat_payload` is the string from `apply_chat_template`.
                    # The processor call for Qwen is: `processor(text=str, images=list_of_pil_images_or_tensor, ...)`
                    # So, `image` should be passed to `images` kwarg.
                    # The `process_vision_info` might be for more complex cases (multiple images, video).
                    # Let's try direct passing first.
                    inputs = active_processor(text=text_prompt_or_chat_payload, images=[image], return_tensors="pt")


                elif "llava" in target_model_id.lower():
                    # `text_prompt_or_chat_payload` is the result of apply_chat_template (a string)
                    # Llava processor takes `text` and `images`
                    inputs = active_processor(text=text_prompt_or_chat_payload, images=image, return_tensors="pt")
                else: # Generic HF model processing
                    inputs = active_processor(text=text_prompt_or_chat_payload, images=image, return_tensors="pt")

                inputs = None
                # Note: `effective_device` here is where the model is (or starts, for device_map).
                # Inputs should be moved to this device.
                if "Qwen" in target_model_id:
                    inputs = active_processor(text=text_prompt_or_chat_payload, images=[image], return_tensors="pt")
                elif "llava" in target_model_id.lower():
                    inputs = active_processor(text=text_prompt_or_chat_payload, images=image, return_tensors="pt")
                else:
                    inputs = active_processor(text=text_prompt_or_chat_payload, images=image, return_tensors="pt")

                inputs = {k: v.to(effective_device) for k, v in inputs.items()}

                with torch.no_grad():
                    generate_ids = active_model.generate(**inputs, max_new_tokens=final_max_new_tokens)

                # Decoding
                if "Qwen" in target_model_id and 'input_ids' in inputs:
                    generated_ids_trimmed = [ out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generate_ids) ]
                    generated_text = active_processor.batch_decode( generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False )[0]
                else:
                    generated_text = active_processor.decode( generate_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True )
            else:
                # This case implies either model_type is not 'hf' or 'ollama', or prerequisites (processor, model) are missing.
                # Or Ollama is disabled.
                if model_type == "ollama" and not settings.OLLAMA_ENABLED:
                    logging.error(f"Ollama is disabled in settings. Cannot process with Ollama model {target_model_id}")
                    raise RuntimeError(f"Ollama is disabled. Cannot process model {target_model_id}.")
                else:
                    logging.error(f"Model {target_model_id} (type: {model_type}) could not be processed. Active model: {active_model is not None}, Active processor: {active_processor is not None}")
                    raise RuntimeError(f"Model {target_model_id} could not be processed due to missing components or unknown type.")

            return self.process_output_text(generated_text)

        except Exception as e:
            logging.error(f"Error during image_to_text for model '{target_model_id}': {e}", exc_info=True)
            if self.ollama_instance and settings.OLLAMA_ENABLED and \
               model_type != "ollama" and \
               (self.force_ollama_for_all or (self.use_ollama_fallback_for and target_model_id in self.use_ollama_fallback_for)):

                logging.warning(f"Hugging Face inference failed for {target_model_id}. Attempting Ollama fallback (model type was {model_type}). Error: {e}")
                _clear_gpu_memory()
                try:
                    ollama_model_to_use = settings.OLLAMA_MODEL_FOR_FALLBACK # Default fallback
                    if target_model_id == "meta-llama/Llama-3.2-11B-Vision-Instruct":
                        ollama_model_to_use = settings.OLLAMA_LLAMA3_2_VISION_MODEL
                    elif "llava" in target_model_id.lower():
                         ollama_model_to_use = "llava"

                    logging.info(f"Using Ollama model for fallback: {ollama_model_to_use}")
                    if hasattr(self.ollama_instance, 'set_model'):
                        self.ollama_instance.set_model(ollama_model_to_use)
                    # else:
                        # Consider re-initializing ollama_instance if model needs to change and no set_model
                        # self.ollama_instance = OllamaProcess(model=ollama_model_to_use, host=settings.OLLAMA_BASE_URL)


                    system_prompt_ollama = self.make_input_prompt.__doc__.splitlines()[1].strip() # Get system prompt from docstring or define centrally
                    pil_image_for_ollama = self.check_image(image_or_file) # Ensure it's a PIL image
                    ollama_prompt_payload = self.ollama_instance.make_prompt(prompt=question, image=pil_image_for_ollama, system_prompt=system_prompt_ollama)
                    generated_text = self.ollama_instance.predict_model(ollama_prompt_payload, max_new_tokens=final_max_new_tokens)

                    # Update cache to reflect that this model_id is now served by Ollama
                    self.models[target_model_id] = (self.ollama_instance, None, "ollama")
                    self.current_model = self.ollama_instance
                    self.current_processor = None
                    self.current_model_id = target_model_id

                    logging.info(f"Ollama fallback successful for {target_model_id} with Ollama model {ollama_model_to_use}")
                    return self.process_output_text(generated_text)
                except Exception as ollama_fallback_e:
                    logging.error(f"Ollama fallback also failed for {target_model_id} (Ollama model {ollama_model_to_use}): {ollama_fallback_e}", exc_info=True)
                    # Re-raise with context of both errors
                    raise RuntimeError(f"HF Error: {e}. Ollama Fallback Error: {ollama_fallback_e}") from ollama_fallback_e

            raise # Re-raise original error if no fallback or fallback failed
        finally:
            # Strategic memory clearing:
            # If a large HF model was used and it's on CUDA, consider clearing.
            # Device_map models are harder to judge for full clearance here.
            if model_type == "hf" and active_model and hasattr(active_model, 'device') and active_model.device.type == 'cuda':
                if not getattr(active_model, 'hf_device_map', None): # Not using device_map (single GPU model)
                    logging.debug(f"Clearing GPU memory after HF model {target_model_id} processing on {active_model.device}")
                    _clear_gpu_memory()
