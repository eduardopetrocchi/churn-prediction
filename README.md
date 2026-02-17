# 🔮 Churn Prediction - MLOps Pipeline

Sistema de predição de churn de clientes utilizando Deep Learning com infraestrutura MLOps completa.

## 📋 Sobre o Projeto

Modelo de rede neural para prever churn de clientes de telecomunicações, com pipeline MLOps completo incluindo versionamento de dados, experimentos e deploy containerizado.

## 🏗️ Arquitetura
```
RNA_CHURN/
├── app/                        # API FastAPI
│   ├── api.py
│   └── templates/
│       ├── index.html
│       ├── form.html
│       ├── upload.html
│       └── result.html
├── data/
│   ├── raw/                    # Dados brutos (DVC)
│   └── processed/              # Dados processados (DVC)
├── metrics/                    # Métricas e plots
│   └── plots/
├── models/                     # Modelos treinados (DVC)
├── notebooks/                  # Análise exploratória
├── src/
│   ├── data/
│   │   └── prepare.py          # Preprocessamento
│   └── model/
│       ├── model_training.py   # Arquitetura da rede neural
│       ├── train.py            # Treinamento + MLflow
│       ├── evaluation.py       # Avaliação + métricas
│       └── predict.py          # Inferência
├── .dvcignore
├── .dockerignore
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.mlflow
├── dvc.yaml                    # Pipeline DVC
├── main.py                     # Pipeline completo
├── params.yaml                 # Hiperparâmetros
└── pyproject.toml
```

## 🛠️ Tecnologias

| Categoria | Tecnologia |
|-----------|------------|
| Deep Learning | TensorFlow / Keras |
| Experiment Tracking | MLflow + DagsHub |
| Data Versioning | DVC |
| API | FastAPI + Uvicorn |
| Containerização | Docker + Docker Compose |
| Linguagem | Python 3.10 |

## ⚙️ Instalação

### 1. Clonar o repositório
```bash
git clone https://github.com/eduardopetrocchi/churn-prediction
cd churn-prediction
```

### 2. Criar ambiente virtual
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Instalar dependências
```bash
pip install -e .
```

### 4. Configurar variáveis de ambiente
```bash
cp .env.example .env
# Editar .env com suas credenciais
```

### 5. Baixar dados com DVC
```bash
dvc pull
```

## 🚀 Execução

### Pipeline Completo
```bash
python -m main
```

### Passo a passo
```bash
# 1. Preprocessar dados
python -m src.data.prepare

# 2. Treinar modelo
python -m src.model.train

# 3. Avaliar modelo
python -m src.model.evaluation

# 4. Subir API
python -m app.api
```

### Via DVC
```bash
dvc repro
```

### Via Docker
```bash
docker-compose up -d --build
```

## 🌐 API

Após subir a API, acesse:

| Endpoint | Descrição |
|----------|-----------|
| `http://localhost:8000` | Página inicial |
| `http://localhost:8000/form` | Predição individual |
| `http://localhost:8000/upload` | Predição em lote (CSV) |

## 🧪 Experimentos DVC
```bash
# Rodar experimento com parâmetros diferentes
dvc exp run -n "exp-lr-001" --set-param model.learning_rate=0.0001

# Comparar experimentos
dvc exp show

# Aplicar melhor experimento
dvc exp apply <exp-name>
```

## 📊 MLflow / DagsHub

Experimentos disponíveis em:
```
https://dagshub.com/eduardopetrocchi/churn-prediction
```

### Métricas rastreadas:
- `test_accuracy`
- `test_precision_weighted`
- `test_recall_weighted`
- `test_f1_weighted`
- `test_roc_auc`

## 📁 Dados

| Arquivo | Descrição |
|---------|-----------|
| `data/raw/telco.csv` | Dataset bruto de telecomunicações |
| `data/processed/train.csv` | Dados de treino processados |
| `data/processed/test.csv` | Dados de teste processados |

## 🧠 Modelo

Rede neural sequencial com:
- Camadas Dense com ativação ReLU
- Dropout para regularização
- Saída com ativação Sigmoid (classificação binária)
- Otimizador Adam
- Loss: Binary Crossentropy

### Hiperparâmetros (`params.yaml`):
```yaml
model:
  hidden_layers: [64, 32]
  dropout: 0.2
  learning_rate: 0.001
  epochs: 50
  batch_size: 64
  patience: 10
```

## 🐳 Docker
```bash
# Build e subir todos os serviços
docker-compose up -d --build

# Serviços disponíveis:
# API:    http://localhost:8000
# MLflow: http://localhost:5000
```

## 📈 Resultados

| Métrica | Valor |
|---------|-------|
| Accuracy | - |
| Precision | - |
| Recall | - |
| F1-Score | - |
| ROC-AUC | - |

> Atualizado após cada execução do pipeline

## 🗂️ Versionamento

| Ferramenta | O que versiona |
|------------|----------------|
| Git | Código, configurações |
| DVC | Dados, modelos |
| MLflow | Experimentos, métricas |

## 🤖 Assistência de IA

Partes do desenvolvimento contaram com assistência do [Claude](https://claude.ai) (Anthropic):
- Templates HTML da interface web
- Correções e formatações no código

## 👤 Autor

**Eduardo Petrocchi**
- DagsHub: [@eduardopetrocchi](https://dagshub.com/eduardopetrocchi)