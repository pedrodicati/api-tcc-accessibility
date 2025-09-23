"""
FastAPI accessibility application main module.

This module provides the main FastAPI application for image and audio processing
to assist people with visual impairments. It includes endpoints for analyzing
images and configuring AI models.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import analyze_image, set_model
from src.config import logger
from src.middleware import SaveRequestResponseMiddleware
from src.process_audio import AudioProcess
from src.process_image import ImageProcess

log = logger()
load_dotenv()


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan context manager for FastAPI.
    
    Handles startup and shutdown events for the FastAPI application,
    including loading AI models and managing resources.
    
    Parameters:
    -----------
    application : FastAPI
        The FastAPI application instance.
        
    Yields:
    -------
    None
        Yields control back to the application after startup.
    """
    try:
        logging.info("Server Started")

        application.state.audio_processor = AudioProcess(
            default_model_id="openai/whisper-small"
        )
        # Model options:
        # - Qwen/Qwen2.5-VL-7B-Instruct
        # - llava-hf/llava-v1.6-mistral-7b-hf
        # - meta-llama/Llama-3.2-11B-Vision-Instruct
        application.state.image_processor = ImageProcess(
            model_id="Qwen/Qwen2.5-VL-7B-Instruct"
        )

        logging.info("Modelos carregados e armazenados em app.state.")

        yield
    finally:
        logging.info("Server Stopped")


app = FastAPI(
    title="Question Image Upload",
    # dependencies=[Depends()] # add auth dependencies here,
    version="0.1.0",
    lifespan=lifespan,
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
app.add_middleware(SaveRequestResponseMiddleware)

app.include_router(analyze_image.router, prefix="/api", tags=["Image/Audio Analysis"])
app.include_router(set_model.router, prefix="/api", tags=["Model Configuration"])

if __name__ == "__main__":
    port: int = int(os.environ.get("PORT", "3015"))
    uvicorn.run(app, host="0.0.0.0", port=port)
