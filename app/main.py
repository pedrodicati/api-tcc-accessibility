import uvicorn
import logging
import os

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from routers import analyze_image
from src.config import logger
from contextlib import asynccontextmanager

from src.process_audio import AudioProcess
from src.process_image import ImageProcess

log = logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logging.info("Server Started")

        app.state.audio_processor = AudioProcess(model_id="openai/whisper-small")
        app.state.image_processor = ImageProcess(
            default_model_id="llava-hf/llava-v1.6-mistral-7b-hf"
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

app.include_router(analyze_image.router, prefix="/api", tags=["Image/Audio Analysis"])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
