from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from src.process_audio import AudioProcess
from src.process_image import ImageProcess

router = APIRouter()


@router.post("/analyze-image-audio-query")
async def create_question_image(
    request: Request,
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
) -> dict:
    try:
        image_content = await image.read()
        audio_content = await audio.read()

        audio_processor: AudioProcess = request.app.state.audio_processor
        image_processor: ImageProcess = request.app.state.image_processor

        # Transcreve o áudio
        transcribed_audio = audio_processor.transcribe(audio_content).get("text", "")

        # Descreve a imagem com base no texto transcrito
        image_text = image_processor.image_to_text(
            image_or_file=image_content,
            question=transcribed_audio,
        )

        # todo: salvar imagem, audio e texto em disco para análise
        # da pra fazer via middleware, ver depois

        return {
            "transcribed_audio": transcribed_audio,
            "image_text": image_text,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
