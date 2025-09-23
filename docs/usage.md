# Guia de Uso

## Executar a API
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 3015
```

## Endpoints
### POST /api/analyze-image-audio-query
Envia imagem + áudio e retorna a transcrição e a descrição da imagem.

```bash
curl -X POST \
  -F "image=@assets/images/almoco.jpg" \
  -F "audio=@assets/audio/audio-test2.m4a" \
  http://localhost:3015/api/analyze-image-audio-query
```

Resposta (exemplo):
```json
{
  "transcribed_audio": "o que tem na imagem?",
  "image_text": "Uma mesa com pratos e talheres..."
}
```

Observação: o middleware salva `image`, `audio` e `response.json` em `processed_files/<timestamp>/`.

### POST /api/set_model
Troca o modelo de visão em runtime.

Campos:
- `new_model_id` (ex.: `Qwen/Qwen2.5-VL-7B-Instruct`, `llava-hf/llava-v1.6-mistral-7b-hf`)
- `model_type` (ex.: `image-text-to-text`)

```bash
curl -X POST \
  -F "new_model_id=Qwen/Qwen2.5-VL-7B-Instruct" \
  -F "model_type=image-text-to-text" \
  http://localhost:3015/api/set_model
```
