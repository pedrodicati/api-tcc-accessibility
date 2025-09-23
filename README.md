# API TCC Accessibility

## Descrição
API em FastAPI para acessibilidade que processa imagem e áudio para gerar descrições de cena orientadas a pessoas com deficiência visual. A API:
- Transcreve um áudio com a pergunta do usuário (Whisper/transformers)
- Usa a pergunta transcrita para descrever a imagem com modelos multimodais (Qwen2.5-VL, LLaVA, ou via Ollama para Llama 3.2 Vision)
- Salva automaticamente imagem, áudio e resposta da API em disco via middleware para auditoria

Esta base inclui qualidade de código (pylint), tipagem e docstrings, além de um fluxo de CI para linting.

## Funcionalidades
- Upload de imagem e áudio (multipart/form-data)
- Transcrição de áudio (Hugging Face transformers)
- Descrição de imagem com modelos vision-language
- Troca dinâmica do modelo de visão em runtime
- Middleware que salva requisição e resposta para análise
- Documentação modular em `/docs`

## Tecnologias Utilizadas
- Python 3.10+
- FastAPI, Uvicorn
- transformers, torch, pillow
- qwen-vl-utils, bitsandbytes (opcional em GPU)
- Ollama (opcional para Llama 3.2 Vision)
- GitHub Actions (pylint)

## Instalação
Resumo rápido. Detalhes em [`docs/installation.md`](docs/installation.md).

```bash
# Windows PowerShell
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
# (opcional para testes/avaliação)
pip install -r tests/requirements.txt
```

## Uso
Resumo. Detalhes e exemplos em [`docs/usage.md`](docs/usage.md) e API em [`docs/api.md`](docs/api.md).

```bash
# Rodar a API (porta padrão 3015)
python -m uvicorn app.main:app --host 0.0.0.0 --port 3015
```

Endpoints principais:
- `POST /api/analyze-image-audio-query` — multipart com `image` e `audio`
- `POST /api/set_model` — troca o modelo de visão (`new_model_id`, `model_type`)

O middleware salva arquivos em `processed_files/<timestamp>/` (imagem, áudio, response.json).

## Exemplos
Veja chamadas `curl` e exemplos de código em [`docs/usage.md`](docs/usage.md).

## Configuração
Variáveis e opções em [`docs/configuration.md`](docs/configuration.md).

Principais variáveis:
- `PORT` (padrão: 3015)
- Modelo de áudio padrão: `openai/whisper-small`
- Modelo de visão padrão: `Qwen/Qwen2.5-VL-7B-Instruct` (ou LLaVA). Llama 3.2 Vision via Ollama.

## Troubleshooting
Erros comuns e soluções em [`docs/troubleshooting.md`](docs/troubleshooting.md).

## Contribuição
Guias e padrões (lint, estilo, PRs) em [`docs/contributing.md`](docs/contributing.md).

## CI / Qualidade de Código
- Pylint via GitHub Actions (`.github/workflows/pylint.yml`)
- Configuração em `.pylintrc`

## Versionamento e Changelog
- Versão atual: `0.1.0`
- Documente mudanças significativas na seção de changelog dos PRs; futuramente mover para `CHANGELOG.md`.

---

Documentação completa:
- [`docs/installation.md`](docs/installation.md)
- [`docs/usage.md`](docs/usage.md)
- [`docs/api.md`](docs/api.md)
- [`docs/configuration.md`](docs/configuration.md)
- [`docs/deployment.md`](docs/deployment.md)
- [`docs/troubleshooting.md`](docs/troubleshooting.md)
- [`docs/contributing.md`](docs/contributing.md)
