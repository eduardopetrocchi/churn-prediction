import json
import os
from pathlib import Path
import yaml
import mlflow
import mlflow.keras
import pandas as pd
import dagshub
from dotenv import load_dotenv

from src.model.model_training import build_model
from tensorflow.keras.callbacks import EarlyStopping

load_dotenv()


def load_params() -> dict[str, float | int]:
    """
    Carrega as configurações globais do arquivo params.yaml.

    Returns:
        dict: Dicionário contendo hiperparâmetros do modelo e configurações de dados.
    """
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params


def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Carrega os conjuntos de treino e teste persistidos na etapa de processamento.

    Returns:
        tuple: (X_train, X_test, y_train, y_test) prontos para o treinamento.
    """
    data_dir = "data/processed"

    train_data = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_data = pd.read_csv(os.path.join(data_dir, "test.csv"))

    X_train = train_data.drop("Churn", axis=1)
    y_train = train_data["Churn"]

    X_test = test_data.drop("Churn", axis=1)
    y_test = test_data["Churn"]

    return X_train, X_test, y_train, y_test


def setup_mlflow_experiment() -> dict:
    """
    Configura a integração entre DagsHub, MLflow e DVC Experiments.

    Inicializa o rastreamento remoto e define se a execução atual deve ser
    tratada como uma 'Nested Run' (execução aninhada) dentro de um experimento DVC.

    Returns:
        dict: Argumentos extras (parent_run_id, run_name, nested) para mlflow.start_run.
    """
    # Inicializar DagsHub
    dagshub.init(
        repo_owner=os.getenv("MLFLOW_TRACKING_USERNAME"), repo_name="churn-prediction"
    )

    # Configurar experimento MLFlow
    mlflow.set_experiment("churn_prediction")

    # Configurar autolog do Keras
    mlflow.keras.autolog()

    # Verificar se está rodando como experimento DVC
    is_dvc_experiment = os.getenv("DVC_EXP_NAME") is not None
    extra_args = {}

    if is_dvc_experiment:
        # Busca ou cria run pai para experimentos DVC
        runs = mlflow.search_runs(
            experiment_ids=[os.getenv("MLFLOW_EMPERIMENT_ID")],
            filter_string="tags.dvc_exp='True'",
            order_by=["start_time DESC"],
        )

        if runs.empty:
            with mlflow.start_run() as parent_run:
                mlflow.set_tag("dvc_exp", True)
                parent_run_id = parent_run.info.run_id
        else:
            parent_run_id = runs.iloc[0].run_id

        run_name = os.getenv("DVC_EXP_NAME")
        extra_args = {
            "parent_run_id": parent_run_id,
            "run_name": run_name,
            "nested": True,
        }

    return extra_args


def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    params: dict,
) -> tuple:
    """
    Orquestra o ciclo de vida do treinamento da rede neural.

    Gerencia a criação do modelo, o ajuste (fit) com Early Stopping, a avaliação
    final e o log de parâmetros e artefatos no servidor MLflow.

    Args:
        X_train (pd.DataFrame): Matriz de features para treinamento.
        X_test (pd.DataFrame): Matriz de features para validação e teste.
        y_train (pd.Series): Vetor de labels para treinamento.
        y_test (pd.Series): Vetor de labels para validação e teste.
        params (dict): Dicionário completo de parâmetros (data e model).

    Returns:
        tuple: (model, history, test_loss, test_accuracy).
    """
    model_params = params["model"]

    # Configurar MLflow
    extra_args = setup_mlflow_experiment()

    with mlflow.start_run(**extra_args):
        # Logar parâmetros
        mlflow.log_params(model_params)
        mlflow.log_param("test_size", params["data"]["test_size"])
        mlflow.log_param("random_state", params["data"]["random_state"])

        # Logar artefatos de processamento
        preprocessor_path = "models/preprocessor.pkl"
        if os.path.exists(preprocessor_path):
            mlflow.log_artifact(preprocessor_path)

        # Criar o modelo
        print("Criando o modelo neural")
        input_dim = X_train.shape[1]
        model = build_model(input_dim=input_dim, params=model_params)

        print(f"Input dimension: {input_dim}")
        print(f"Model architecture:")
        model.summary()

        # Configurar early stopping
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=model_params.get("patience", 10),
            restore_best_weights=True,
            verbose=1,
        )

        # Treinar o modelo
        print("\nTreinando o modelo...")
        history = model.fit(
            X_train,
            y=y_train,
            validation_data=(X_test, y_test),
            epochs=model_params["epochs"],
            batch_size=model_params["batch_size"],
            callbacks=[early_stopping],
            verbose=1,
        )

        print("\nAvaliando o modelo no conjunto de teste")
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

        print(f"\nResultados finais:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f}")

        # Logar métricas finais
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_accuracy)

        # Salvar o modelo e histórico localmente
        save_training_artifacts(model, history, test_loss, test_accuracy)

        # Logar artefatos finais no MLFlow
        mlflow.log_artifact("models/model.keras")
        mlflow.log_artifact("metrics/training.json")

        return model, history, test_loss, test_accuracy


def save_training_artifacts(
    model, history, test_loss: float, test_accuracy: float
) -> None:
    """
    Persiste o modelo treinado e as métricas de evolução em formato digital.

    Args:
        model (keras.Model): Modelo treinado a ser salvo.
        history (keras.callbacks.History): Histórico do treinamento (loss/accuracy).
        test_loss (float): Loss final calculado no conjunto de teste.
        test_accuracy (float): Acurácia final calculada no conjunto de teste.
    """

    # Criar diretórios
    models_dir = Path("models")
    metrics_dir = Path("metrics")
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Salvar o modelo
    model_path = models_dir / "model.keras"
    print(f"\nSalvando o modelo em: {model_path}")
    model.save(model_path)

    # Salvar histórico de treinamento e métricas finais
    training_metrics = {
        "train_loss": [float(x) for x in history.history["loss"]],
        "train_accuracy": [float(x) for x in history.history["accuracy"]],
        "val_loss": [float(x) for x in history.history["val_loss"]],
        "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
        "final_test_loss": float(test_loss),
        "final_test_accuracy": float(test_accuracy),
        "epochs_trained": len(history.history["loss"]),
    }

    metrics_path = metrics_dir / "training.json"
    print(f"Salvando métricas em: {metrics_path}")
    with open(metrics_path, "w") as f:
        json.dump(training_metrics, f, indent=4)


def main() -> None:
    """
    Ponto de entrada do pipeline de treinamento.

    Orquestra o carregamento de parâmetros e dados, executa o treinamento
    e exibe o sumário final de performance no console.
    """
    print("=" * 60)
    print("INICIANDO TREINAMENTO DO MODELO")
    print("=" * 60)

    # 1. Carregar parâmetros
    print("\n1. Carregando parâmetros...")
    params = load_params()
    print("✓ Parâmetros carregados")

    # 2. Carregar dados processados
    print("\n2. Carregando dados processados...")
    X_train, X_test, y_train, y_test = load_processed_data()
    print(f"✓ Dados carregados:")
    print(f"  - Treino: {X_train.shape[0]} amostras, {X_train.shape[1]} features")
    print(f"  - Teste: {X_test.shape[0]} amostras, {X_test.shape[1]} features")

    # 3. Treinar modelo
    print("\n3. Treinando modelo neural...")
    model, history, test_loss, test_accuracy = train_model(
        X_train, X_test, y_train, y_test, params
    )

    print("\n" + "=" * 60)
    print("TREINAMENTO CONCLUÍDO COM SUCESSO!")
    print("=" * 60)
    print(f"Acurácia final no teste: {test_accuracy:.2%}")
    print(f"Loss final no teste: {test_loss:.4f}")


if __name__ == "__main__":
    main()
