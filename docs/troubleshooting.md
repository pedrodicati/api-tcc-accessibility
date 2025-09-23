# Troubleshooting

## ImportError no CI (pylint E0401)
- Certifique-se que `requirements.txt` e `tests/requirements.txt` são instalados no CI.
- O workflow já foi atualizado para instalar ambos.

## Memória insuficiente ao carregar modelos grandes
- Use quantização 4-bit (já configurado quando há CUDA)
- Prefira o caminho via Ollama para Llama 3.2 Vision

## Arquivos não aparecem em processed_files
- Verifique se a requisição é `multipart/form-data`
- Confirme permissões de escrita do processo
