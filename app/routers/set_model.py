"""
Router for model configuration endpoints.

This module provides endpoints for dynamically changing AI models
used for image processing.
"""

from typing import Dict, Any

import torch
from fastapi import APIRouter, HTTPException, Request

from src.process_image import ImageProcess

router = APIRouter()


@router.post("/set_model")
async def set_model(
    request: Request, new_model_id: str, model_type: str
) -> Dict[str, Any]:
    """
    Change the AI model used for image processing.

    This endpoint allows dynamic switching between different image-to-text
    models during runtime, clearing GPU memory and loading the new model.

    Parameters:
    -----------
    request : Request
        FastAPI request object containing application state.
    new_model_id : str
        The identifier of the new model to load.
    model_type : str
        The type of model being loaded.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing a success message with the new model ID.

    Raises:
    -------
    HTTPException
        If there's an error during model loading or GPU memory management.
    """
    try:
        # clear usage memory gpu
        torch.cuda.empty_cache()

        # Carrega outro modelo, substitui o anterior
        image_processor = ImageProcess(model_id=new_model_id, model_type=model_type)
        request.app.state.image_processor = image_processor

        return {"message": f"Modelo trocado para {new_model_id}"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
