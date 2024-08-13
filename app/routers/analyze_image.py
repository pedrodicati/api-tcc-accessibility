from fastapi import APIRouter, UploadFile, File
from src.process_audio import ProcessAudio

router = APIRouter()
audio_processor = ProcessAudio(model_id="openai/whisper-small")


@router.post("/analyze-image-audio-query")
async def create_question_image(
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
) -> dict:
    image_content = await image.read()
    audio_content = await audio.read()

    transcriptions = audio_processor.transcribe(audio_content)

    return {
        "transcriptions": transcriptions,
    }
