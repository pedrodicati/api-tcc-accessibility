import logging
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from typing import Optional

from src.process_image import ImageProcess
from src.process_audio import AudioProcess # Import AudioProcess
# Import custom exceptions if we want to catch them specifically here,
# otherwise they will propagate to global handlers.
# from app.src.exceptions import ModelLoadError, InvalidInputError # Example

router = APIRouter()

class SetImageModelRequest(BaseModel): # Renamed
    new_model_id: str = Field(..., description="The ID of the new model to set for image processing.")
    load_now: Optional[bool] = Field(True, description="Whether to load the model immediately.")
    # Optional device and dtype overrides for this specific model load
    device_str: Optional[str] = Field(None, description="Optional: Device to load this model on (e.g., 'cuda', 'cpu'). Overrides default.")
    torch_dtype_str: Optional[str] = Field(None, description="Optional: Dtype for this model (e.g., 'float16', 'bfloat16'). Overrides default.")

# Pydantic model for the audio model request
class SetAudioModelRequest(BaseModel):
    new_model_id: str = Field(..., description="The ID of the new model to set for audio processing.")
    load_now: Optional[bool] = Field(True, description="Whether to load the model immediately.")
    device_str: Optional[str] = Field(None, description="Optional: Device to load this model on. Overrides default.")
    torch_dtype_str: Optional[str] = Field(None, description="Optional: Dtype for this model. Overrides default.")


@router.post("/set_image_model") # Renamed route
async def set_image_model_endpoint(request: Request, payload: SetImageModelRequest) -> dict: # Renamed function and type hint
    # No try-except here for custom exceptions; let them propagate to global handlers in main.py
    # FastAPI will automatically handle Pydantic validation errors for SetImageModelRequest,
    # returning a 422 response if the payload is invalid.

    image_processor: ImageProcess = request.app.state.image_processor

    # This check is more of an assertion, as FastAPI should ensure app.state is available
    # if lifespan is correctly managed. If image_processor is None, it's a server setup issue.
    if not isinstance(image_processor, ImageProcess):
        logging.error("Image processor not found or not of correct type in app.state. This indicates a server setup issue.")
        # This situation is a server error, not a client error normally.
        # Raising a generic exception here will be caught by FastAPI's default 500 handler or a custom one if defined.
        raise Exception("Image processor is not configured correctly on the server.")

    image_processor.set_model(
        model_id=payload.new_model_id,
        load_now=payload.load_now,
        device_str=payload.device_str, # Pass through optional device/dtype
        torch_dtype_str=payload.torch_dtype_str
    )

    logging.info(f"Image model set to {payload.new_model_id}. Load now was {payload.load_now}.")
    action = "loaded immediately" if payload.load_now else "set and will load on first use"
    return {"message": f"Image model successfully {action}: {payload.new_model_id}."}


@router.post("/set_audio_model")
async def set_audio_model_endpoint(request: Request, payload: SetAudioModelRequest) -> dict:
    audio_processor: AudioProcess = request.app.state.audio_processor

    if not isinstance(audio_processor, AudioProcess):
        logging.error("Audio processor not found or not of correct type in app.state. This indicates a server setup issue.")
        raise Exception("Audio processor is not configured correctly on the server.")

    # Call the set_model method of AudioProcess
    # This method is expected to return a dict message or raise an exception
    response_message = audio_processor.set_model(
        model_id=payload.new_model_id,
        load_now=payload.load_now,
        device_str=payload.device_str,
        torch_dtype_str=payload.torch_dtype_str
    )

    logging.info(f"Audio model processing: {response_message.get('message', 'No message returned from set_model')}")
    # The message from audio_processor.set_model will be returned.
    # If an error occurs (e.g. ModelLoadError), it will be caught by global handlers.
    return response_message
