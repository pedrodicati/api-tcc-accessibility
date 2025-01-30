import uvicorn
import logging
import os

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from routers import analyze_image, set_model
from src.config import logger
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from src.process_audio import AudioProcess
from src.process_image import ImageProcess

log = logger()
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logging.info("Server Started")

        app.state.audio_processor = AudioProcess(
            default_model_id="openai/whisper-small"
        )
        # Model options:
        # - Qwen/Qwen2.5-VL-7B-Instruct
        # - llava-hf/llava-v1.6-mistral-7b-hf
        # - meta-llama/Llama-3.2-11B-Vision-Instruct
        app.state.image_processor = ImageProcess(model_id="Qwen/Qwen2.5-VL-7B-Instruct")

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

app.include_router(analyze_image.router, prefix="/api", tags=["Image/Audio Analysis"])
app.include_router(set_model.router, prefix="/api", tags=["Model Configuration"])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
