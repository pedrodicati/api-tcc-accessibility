"""
Router for image and audio analysis endpoints.

This module provides endpoints for analyzing images with audio queries
to assist people with visual impairments.
"""

from typing import Dict, Any

from fastapi import APIRouter, UploadFile, File, HTTPException, Request

from src.process_audio import AudioProcess
from src.process_image import ImageProcess

router = APIRouter()


@router.post("/analyze-image-audio-query")
async def create_question_image(
    request: Request,
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
) -> Dict[str, Any]:
    """
    Analyze an image based on an audio query.
    
    This endpoint transcribes audio content to text and uses it as a query
    to generate a description of the uploaded image, helping visually
    impaired users understand visual content.
    
    Parameters:
    -----------
    request : Request
        FastAPI request object containing application state.
    image : UploadFile
        The image file to be analyzed.
    audio : UploadFile
        The audio file containing the user's question.
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the transcribed audio text and image description.
        
    Raises:
    -------
    HTTPException
        If there's an error during image or audio processing.
    """
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

        # TODO: salvar imagem, audio e texto em disco para análise
        # da pra fazer via middleware, ver depois

        return {
            "transcribed_audio": transcribed_audio,
            "image_text": image_text,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
