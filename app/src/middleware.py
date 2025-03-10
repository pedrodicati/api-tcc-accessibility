from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from datetime import datetime
import os
import logging

# Diretório base para salvar os arquivos
SAVE_DIR = "processed_files"
os.makedirs(SAVE_DIR, exist_ok=True)


class SaveRequestResponseMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        logging.info(f"Requisição recebida: {request.url.path}")

        body = await request.body()

        async def receive():
            return {"type": "http.request", "body": body}

        request = Request(request.scope, receive)

        try:
            if "multipart/form-data" in request.headers.get("content-type", ""):
                form = await request.form()
                image = form.get("image")
                audio = form.get("audio")

                logging.info(
                    f"Imagem recebida: {image.filename if image else 'Nenhuma'}"
                )
                logging.info(f"Áudio recebido: {audio.filename if audio else 'Nenhum'}")
            else:
                image = audio = None
                logging.info("Requisição não é do tipo multipart/form-data.")
        except Exception as e:
            logging.error(f"Erro ao processar o formulário: {e}")
            image = audio = None

        response = await call_next(request)

        if image and audio and "/analyze-image-audio-query" in request.url.path:
            logging.info("Salvando arquivos...")

            # Criar a pasta com o timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_path = os.path.join(SAVE_DIR, timestamp)
            os.makedirs(folder_path, exist_ok=True)  # Criação da pasta

            try:
                # Salvar o áudio
                audio_content = await audio.read()
                audio_path = os.path.join(folder_path, "audio.wav")
                with open(audio_path, "wb") as f:
                    f.write(audio_content)
                logging.info(
                    f"Áudio salvo em: {audio_path} (tamanho: {len(audio_content)} bytes)"
                )

                # Salvar a imagem
                image_content = await image.read()
                image_extension = (
                    os.path.splitext(image.filename)[-1] or ".jpg"
                )  # Preservar a extensão original
                image_path = os.path.join(folder_path, f"image{image_extension}")
                with open(image_path, "wb") as f:
                    f.write(image_content)
                logging.info(
                    f"Imagem salva em: {image_path} (tamanho: {len(image_content)} bytes)"
                )

                # Salvar a resposta da API
                response_body = b""
                async for chunk in response.body_iterator:
                    response_body += chunk

                response_path = os.path.join(folder_path, "response.json")
                with open(response_path, "wb") as f:
                    f.write(response_body)
                logging.info(
                    f"Resposta salva em: {response_path} (tamanho: {len(response_body)} bytes)"
                )

                # Recriar a resposta para o cliente
                response = Response(
                    content=response_body,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type,
                )
            except Exception as e:
                logging.error(f"Erro ao salvar os arquivos: {e}")
        else:
            logging.error("Nenhum arquivo ou endpoint incompatível para salvar.")

        return response
