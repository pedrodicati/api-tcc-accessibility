from fastapi import APIRouter, UploadFile, File
from src.process_audio import AudioProcess
from src.process_image import ImageProcess

router = APIRouter()
audio_processor = AudioProcess(model_id="openai/whisper-small")
image_processor = ImageProcess(model_id="llava-hf/llava-1.5-7b-hf")


@router.post("/analyze-image-audio-query")
async def create_question_image(
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
) -> dict:
    image_content = await image.read()
    audio_content = await audio.read()

    transcribed_audio = audio_processor.transcribe(audio_content).get("text", "")

    image_text = image_processor.image_to_text(image_content, transcribed_audio)

    return {
        "transcribed_audio": transcribed_audio,
        "image_text": image_text,
    }
