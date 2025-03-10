# 📚 Pesquisa de Pesquisadores com Hugging Face e Google Scholar

Este projeto permite buscar pesquisadores no Google Scholar e formatar as informações utilizando a API da Hugging Face.

## 🚀 Tecnologias Utilizadas
- Python 3+
- [Scholarly](https://pypi.org/project/scholarly/) (para busca no Google Scholar)
- [Transformers](https://huggingface.co/docs/transformers/index) (para geração de texto)
- Hugging Face API

## 📦 Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/seu-repositorio.git
   cd seu-repositorio
   ```

2. Crie um ambiente virtual (opcional, recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate  # Windows
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

## 🔑 Configuração da API

Antes de executar o projeto, copie o arquivo `.env.example` e renomeie para `.env`:

```bash
cp .env.example .env
```

Depois, edite o arquivo `.env` e adicione sua chave da API da Hugging Face:

```
HUGGINGFACEHUB_API_TOKEN=sua-chave-aqui
```

## ▶️ Como Executar

1. Execute o script:
   ```bash
   python main.py
   ```
2. Digite o tema da pesquisa quando solicitado.
3. O programa buscará os pesquisadores e formatará os resultados com a API da Hugging Face.

## 🛠 Estrutura do Projeto

```
/
├── main.py  # Script principal
├── requirements.txt  # Dependências do projeto
├── .gitignore  # Arquivos ignorados pelo Git
├── .env.example  # Exemplo do arquivo de variáveis de ambiente
├── README.md  # Documentação do projeto
```
