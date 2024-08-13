import uvicorn
import os

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from routers import analyze_image
from src.config import logger

app = FastAPI(
    title="Question Image Upload",
    # dependencies=[Depends()] # add auth dependencies here
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_image.router)

log = logger()

@app.on_event("startup")
async def startup_event():
    log.info("Starting up")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))