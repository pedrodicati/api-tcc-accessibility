from pydantic import BaseModel
from fastapi import File, UploadFile

class PostQuestion(BaseModel):
    image: UploadFile = File(..., media_type="image/jpg")
    question_audio: UploadFile = File(...)