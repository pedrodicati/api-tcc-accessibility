# Documentação da API

## Base URL
```
http://localhost:3015
```

## POST /api/analyze-image-audio-query
Analisa imagem com base em uma pergunta vinda do áudio.

- Content-Type: `multipart/form-data`
- Campos:
  - `image`: arquivo de imagem (jpg/png)
  - `audio`: arquivo de áudio com a pergunta (m4a/wav)

Resposta:
```json
{
  "transcribed_audio": "string",
  "image_text": "string"
}
```

Códigos:
- 200 OK
- 400/422 Erro de validação
- 500 Erro interno

## POST /api/set_model
Troca o modelo de visão usado pela API.

- Content-Type: `multipart/form-data` ou `application/x-www-form-urlencoded`
- Campos:
  - `new_model_id`: string (ex.: `Qwen/Qwen2.5-VL-7B-Instruct`)
  - `model_type`: string (ex.: `image-text-to-text`)

Resposta:
```json
{ "message": "Modelo trocado para <new_model_id>" }
```

Códigos:
- 200 OK
- 500 Erro interno
