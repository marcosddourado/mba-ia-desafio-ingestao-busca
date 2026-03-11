# Desafio MBA Engenharia de Software com IA - Full Cycle

Solução de ingestão de PDF e busca semântica via CLI usando LangChain, OpenAI e PostgreSQL com pgVector.

## Como funciona

1. **Ingestão** (`ingest.py`): lê o `document.pdf`, divide em chunks de 1000 caracteres (overlap 150), gera embeddings com OpenAI e armazena no PostgreSQL com pgVector.
2. **Chat** (`chat.py`): recebe perguntas do usuário via terminal, busca os 10 chunks mais relevantes no banco vetorial e envia o contexto para o modelo `gpt-5-nano` responder com base apenas no conteúdo do PDF.

## Pré-requisitos

- Python 3.10+
- Docker e Docker Compose
- Chave de API da OpenAI

## Configuração

### 1. Ambiente virtual

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Variáveis de ambiente

Copie o arquivo de exemplo e preencha com suas credenciais:

```bash
cp .env.example .env
```

## Ordem de execução

### 1. Subir o banco de dados

```bash
docker compose up -d
```

### 2. Executar a ingestão do PDF

```bash
python src/ingest.py
```

### 3. Iniciar o chat

```bash
python src/chat.py
```

## Exemplo de uso

```
Chat iniciado. Digite 'sair' para encerrar.

PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.

PERGUNTA: sair
Encerrando o chat.
```

Perguntas fora do conteúdo do PDF retornam: *"Não tenho informações necessárias para responder sua pergunta."*
