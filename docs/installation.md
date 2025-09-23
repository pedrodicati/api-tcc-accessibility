# Instalação

## Requisitos
- Python 3.10+
- Windows / macOS / Linux
- (Opcional) GPU com CUDA para aceleração de modelos

## Passos
```bash
# 1) Criar e ativar ambiente virtual (Windows PowerShell)
python -m venv .venv
. .venv/Scripts/Activate.ps1

# 2) Atualizar pip
pip install --upgrade pip

# 3) Instalar dependências
pip install -r requirements.txt

# 4) (Opcional) Dependências de testes/avaliação
pip install -r tests/requirements.txt
```

## Ollama (opcional, para Llama 3.2 Vision)
- Instale o Ollama conforme a documentação oficial
- Garanta que o comando `ollama` esteja disponível no PATH
- O código fará o `pull` do modelo automaticamente quando necessário

## Verificação
```bash
python -c "import fastapi, torch, transformers, PIL; print('ok')"
```
