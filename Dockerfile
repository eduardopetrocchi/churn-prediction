# Usar imagem base Python
FROM python:3.10-slim

# Definir diretório de trabalho
WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt get install -y 

# Copiar arquivos de dependências
COPY pyproject.toml ./ 

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