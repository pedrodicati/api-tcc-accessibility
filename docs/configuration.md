# Configuração

## Variáveis de Ambiente
- `PORT`: Porta do servidor (padrão `3015`)

## Modelos
- Áudio (ASR): padrão `openai/whisper-small`
- Visão (VLM): padrão `Qwen/Qwen2.5-VL-7B-Instruct`
- Alternativas: `llava-hf/llava-v1.6-mistral-7b-hf`, `meta-llama/Llama-3.2-11B-Vision-Instruct` (via Ollama)

## GPU e Quantização
- Se CUDA disponível, ativa `BitsAndBytesConfig` (4bit) para reduzir memória
- `torch_dtype` ajustado conforme dispositivo

## Salvamento de Arquivos
- Middleware salva em `processed_files/<timestamp>/`:
  - `audio.wav`
  - `image.<ext>`
  - `response.json`
