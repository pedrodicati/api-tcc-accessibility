# Deployment

## Uvicorn/Gunicorn
```bash
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-3015}
```

Para produção, considere usar um gerenciador (systemd, PM2, supervisor) e um proxy reverso (Nginx).

## Docker (exemplo)
- Ajuste `Dockerfile` conforme o ambiente (CPU/GPU)
- Use variáveis de ambiente para porta e modelos

```bash
docker build -t api-tcc-accessibility .
docker run -p 3015:3015 --env PORT=3015 api-tcc-accessibility
```

