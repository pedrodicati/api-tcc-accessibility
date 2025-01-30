import torch
from fastapi import APIRouter, HTTPException, Request
from src.process_image import ImageProcess

router = APIRouter()


@router.post("/set_model")
async def set_model(request: Request, new_model_id: str, model_type: str) -> dict:
    try:
        # clear usage memory gpu
        torch.cuda.empty_cache()

        # Carrega outro modelo, substitui o anterior
        image_processor = ImageProcess(
            default_model_id=new_model_id, model_type=model_type
        )
        request.app.state.image_processor = image_processor

        return {"message": f"Modelo trocado para {new_model_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
