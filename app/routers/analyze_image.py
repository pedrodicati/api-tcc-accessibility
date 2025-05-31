from fastapi import APIRouter, UploadFile, File, Request
import logging
import os # Added
import json # Added
from datetime import datetime # Added

from src.process_audio import AudioProcess
from src.process_image import ImageProcess
from app.src.settings import settings # Added
# from app.src.exceptions import FileProcessingError, InferenceError # Example

router = APIRouter()
log = logging.getLogger(__name__) # Get specific logger instance for this router


@router.post("/analyze-image-audio-query")
async def create_question_image(
    request: Request,
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
    # Potentially add other form fields here if needed, then a Pydantic model could consume them
    # e.g., custom_max_tokens: Optional[int] = Form(None)
) -> dict:

    # Read file contents first as processing functions might consume the file pointer
    # This is important because we need to save the original files after processing.
    # Need to reset file pointers before reading for saving if they are read by processors.
    # The processors (check_audio, check_image) already read the content into bytes/PIL Image.
    # So we should pass the UploadFile objects and let the processors read them.
    # Then, for saving, we will need to seek(0) on the UploadFile objects.

    audio_processor: AudioProcess = request.app.state.audio_processor
    image_processor: ImageProcess = request.app.state.image_processor

    if not audio_processor or not image_processor:
        log.critical("Audio or Image processor not initialized in app state. Server configuration error.")
        raise Exception("Server not configured correctly: processors missing.")

    # Process audio and image
    # The `transcribe` and `image_to_text` methods should handle internal errors and raise custom exceptions.
    # `check_audio` and `check_image` inside them will raise FileProcessingError for bad files.

    # Pass UploadFile directly; processors will call .read()
    transcription_result = audio_processor.transcribe(audio.file) # Pass the SpooledTemporaryFile
    transcribed_audio = transcription_result.get("text", "")

    image_text = image_processor.image_to_text(
        image_or_file=image.file, # Pass the SpooledTemporaryFile
        question=transcribed_audio,
    )

    api_response = {
        "transcribed_audio": transcribed_audio,
        "image_text": image_text,
    }

    # Save files if enabled
    if settings.SAVE_ANALYSIS_FILES:
        try:
            # Create a unique directory for this request
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            save_dir = os.path.join(settings.ANALYSIS_FILES_SAVE_DIR, timestamp)
            os.makedirs(save_dir, exist_ok=True)
            log.info(f"Saving analysis files to: {save_dir}")

            # Save original audio file
            audio_filename = audio.filename if audio.filename else "uploaded_audio.dat"
            audio_path = os.path.join(save_dir, os.path.basename(audio_filename))
            audio.file.seek(0) # Use synchronous seek for SpooledTemporaryFile
            with open(audio_path, "wb") as f_audio:
                # Read synchronously in chunks
                while chunk := audio.file.read(8192):
                    f_audio.write(chunk)
            log.info(f"Saved audio file: {audio_path}")

            # Save original image file
            image_filename = image.filename if image.filename else "uploaded_image.dat"
            image_path = os.path.join(save_dir, os.path.basename(image_filename))
            image.file.seek(0) # Use synchronous seek
            with open(image_path, "wb") as f_image:
                # Read synchronously in chunks
                while chunk := image.file.read(8192):
                    f_image.write(chunk)
            log.info(f"Saved image file: {image_path}")

            # Save API response as JSON
            response_json_path = os.path.join(save_dir, "response.json")
            with open(response_json_path, "w", encoding="utf-8") as f_json:
                json.dump(api_response, f_json, ensure_ascii=False, indent=4)
            log.info(f"Saved API response: {response_json_path}")

        except Exception as e:
            log.error(f"Failed to save analysis files for request {timestamp}: {e}", exc_info=True)
            # Do not re-raise; failing to save should not fail the main API response.

    return api_response
