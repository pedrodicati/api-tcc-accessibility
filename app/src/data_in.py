from pydantic import BaseModel
from fastapi import File, UploadFile


class PostAnalyzeImageQuery(BaseModel):
    image: UploadFile = File(...)
    audio: UploadFile = File(...)