Para evitar que as bibliotecas utilizadas para avaliar os modelos impactem o ambiente de desenvolvimento, foi criado um ambiente virtual separado para a execução dos testes. Para isso, foi utilizado o pacote `venv` do Python, que cria um ambiente virtual isolado. Para criar o ambiente virtual, garanta que você está dentro do diretório `tests` e execute o seguinte comando:

```bash
python -m venv .venv
```

Após a criação do ambiente virtual, ative-o com o comando:

```bash
source .venv/bin/activate
```

Com o ambiente virtual ativado, instale as dependências necess

```bash
pip install -r requirements.txt
```

Os testes estão relacionados com o benchmark dos modelos testados nesta API.