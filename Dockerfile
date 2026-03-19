# Usar imagem base Python
FROM python:3.10-slim

# Definir diretório de trabalho
WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copiar arquivos de dependências primeiro (para aproveitar cache do Docker)
COPY pyproject.toml ./

# Instalar o pacote em modo editável e suas dependências
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .

# Copia código fonte
COPY . .    

# Criar diretórios necessários
RUN mkdir -p data/raw data/processed models metrics/plots predictions

# Expor porta da API
EXPOSE 8000

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Comando padrão (API)
CMD [ "python", "-m", "app.api" ]