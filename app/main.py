import uvicorn
import logging
import os

from fastapi import FastAPI, Depends, Request, status # Added Request, status
from fastapi.responses import JSONResponse # Added JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from routers import analyze_image, set_model
# Import custom exceptions
from src.exceptions import (
    BaseAppException,
    ModelNotFoundError,
    ModelLoadError,
    InferenceError,
    InvalidInputError,
    OllamaNotAvailableError,
    FileProcessingError
)
from src.config import logger # Will be updated to use settings later
from src.settings import settings # Import settings
from contextlib import asynccontextmanager
from dotenv import load_dotenv # Pydantic-settings handles .env, but this might be used elsewhere or can be removed if not.

from src.process_audio import AudioProcess
from src.process_image import ImageProcess
from src.ollama_process import OllamaProcess
# Middleware import removed: from src.middleware import SaveRequestResponseMiddleware

# log = logger() # Logger initialization might be better after settings are loaded, or logger itself uses settings.
# For now, assume logger() in config.py will be adapted.
load_dotenv() # Pydantic-settings loads .env by default if configured, this might be redundant.
             # Keeping it for now in case other parts of the app use it directly.


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize logger instance after settings are available if logger depends on settings
    log = logger() # Now logger() from config.py can use settings.LOG_LEVEL
    try:
        log.info("Server Startup: Initializing processors...")

        app.state.audio_processor = AudioProcess(
            default_model_id=settings.DEFAULT_AUDIO_MODEL_ID,
            default_device=settings.DEFAULT_DEVICE,
            default_torch_dtype_str=settings.DEFAULT_TORCH_DTYPE_STR
        )

        ollama_processor = None
        if settings.OLLAMA_ENABLED:
            try:
                # Assuming OllamaProcess constructor takes host and potentially a default model.
                # Adjust if OllamaProcess has a different configuration mechanism.
                ollama_processor = OllamaProcess(
                    host=settings.OLLAMA_BASE_URL,
                    model=settings.OLLAMA_MODEL_FOR_FALLBACK # Default model for the instance
                )
                log.info(f"OllamaProcess initialized with host {settings.OLLAMA_BASE_URL} and default model {settings.OLLAMA_MODEL_FOR_FALLBACK}")
            except Exception as e:
                log.error(f"Failed to initialize OllamaProcess: {e}. Ollama fallback will be disabled.", exc_info=True)
                # Fallback to None if Ollama initialization fails
                ollama_processor = None
        else:
            log.info("Ollama is disabled by settings. OllamaProcess not initialized.")


        app.state.image_processor = ImageProcess(
            default_model_id=settings.DEFAULT_IMAGE_MODEL_ID,
            ollama_instance=ollama_processor, # Pass the configured (or None) instance
            default_device=settings.DEFAULT_DEVICE,
            default_torch_dtype_str=settings.DEFAULT_TORCH_DTYPE_STR
            # force_ollama_for_all can be another setting if needed e.g. settings.IMAGE_FORCE_OLLAMA
        )

        log.info("Image and Audio processors initialized and stored in app.state.")

        yield
    finally:
        log.info("Server Shutdown")


app = FastAPI(
    title="Question Image Upload",
    # dependencies=[Depends()] # add auth dependencies here,
    version="0.1.0",
    lifespan=lifespan,
)

# Exception Handlers
@app.exception_handler(ModelNotFoundError)
async def model_not_found_exception_handler(request: Request, exc: ModelNotFoundError):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"message": f"Model not found: {exc.message}"},
    )

@app.exception_handler(ModelLoadError)
async def model_load_exception_handler(request: Request, exc: ModelLoadError):
    # Log the full error for server-side diagnosis if needed
    logging.error(f"ModelLoadError caught: {exc.message}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE, # 503 might be suitable as model loading is a setup/service issue
        content={"message": f"Failed to load model: {exc.message}. Please try again later or contact support."},
    )

@app.exception_handler(InferenceError)
async def inference_exception_handler(request: Request, exc: InferenceError):
    logging.error(f"InferenceError caught: {exc.message}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"message": f"Error during inference: {exc.message}."},
    )

@app.exception_handler(InvalidInputError)
async def invalid_input_exception_handler(request: Request, exc: InvalidInputError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"message": f"Invalid input: {exc.message}"},
    )

@app.exception_handler(OllamaNotAvailableError)
async def ollama_not_available_exception_handler(request: Request, exc: OllamaNotAvailableError):
    logging.warning(f"OllamaNotAvailableError caught: {exc.message}") # Warning, as it might be a config issue
    return JSONResponse(
        status_code=status.HTTP_501_NOT_IMPLEMENTED, # Or 503 if Ollama is seen as a temporarily unavailable service
        content={"message": f"Ollama operation failed: {exc.message}."},
    )

@app.exception_handler(FileProcessingError)
async def file_processing_exception_handler(request: Request, exc: FileProcessingError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST, # Usually input file issues
        content={"message": f"File processing error: {exc.message}"},
    )

# Generic handler for BaseAppException if any other custom exception inherits from it and is not caught explicitly
@app.exception_handler(BaseAppException)
async def base_app_exception_handler(request: Request, exc: BaseAppException):
    logging.error(f"BaseAppException caught: {exc.message}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"message": f"An application error occurred: {exc.message}"},
    )


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware para salvar os arquivos de imagem, áudio e resposta da API
# app.add_middleware(SaveRequestResponseMiddleware) # This line is removed

app.include_router(analyze_image.router, prefix="/api", tags=["Image/Audio Analysis"])
app.include_router(set_model.router, prefix="/api", tags=["Model Configuration"])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 3015)))
