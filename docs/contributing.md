# Contribuição

## Padrões
- Python 3.10+
- Tipagem e docstrings em funções/classes
- Lint: `pylint` com `.pylintrc`
- Estilo: mantenha linhas até 100 colunas

## Workflow
1. Crie um branch a partir de `main`
2. Rode lint localmente
   ```bash
   pylint $(git ls-files '*.py')
   ```
3. Abra PR com descrição clara (inclua mudanças no README/docs se aplicável)

## Commits
- Use mensagens descritivas (ex.: `feat: ...`, `fix: ...`, `docs: ...`)
- Referencie issues quando aplicável
