import json
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from keras.models import load_model as keras_load_model
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


def load_params() -> dict:
    """
    Carrega o arquivo completo de parâmetros do projeto.

    Returns:
        dict: Configurações carregadas do 'params.yaml'.
    """
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params


def load_model() -> object:
    """
    Carrega o modelo Keras pré-treinado do diretório de modelos.

    Returns:
        keras.Model: O modelo de rede neural carregado.
    """
    model_path = "models/model.keras"
    print(f"Carregando modelo de: {model_path}")
    model = keras_load_model(model_path)
    return model


def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Lê os dados de teste processados e separa as features do target.

    Returns:
        tuple[pd.DataFrame, pd.Series]: DataFrame X_test e Series y_test.
    """
    data_dir = "data/processed"
    test_data = pd.read_csv(os.path.join(data_dir, "test.csv"))

    X_test = test_data.drop("Churn", axis=1)
    y_test = test_data["Churn"]

    return X_test, y_test


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Calcula as métricas de performance do modelo nos dados de teste.

    Gera predições de probabilidade e classe, calculando métricas como 
    F1-Score, ROC-AUC e Matriz de Confusão.

    Args:
        model (keras.Model): Modelo treinado a ser avaliado.
        X_test (pd.DataFrame): Features de teste processadas.
        y_test (pd.Series): Rótulos reais de teste.

    Returns:
        dict: Dicionário contendo métricas escalares e os componentes da matriz de confusão.
    """
    print("Avaliando o modelo...")

    # Predições
    y_pred_proba = model.predict(X_test).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Métricas
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": float(report["accuracy"]),
        "precision_weighted": float(report["weighted avg"]["precision"]),
        "recall_weighted": float(report["weighted avg"]["recall"]),
        "f1_weighted": float(report["weighted avg"]["f1-score"]),
        "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
    }

    return metrics


def print_results(metrics: dict) -> None:
    """
    Exibe os resultados da avaliação de forma formatada no terminal.

    Args:
        metrics (dict): Dicionário de métricas retornado por evaluate_model.
    """
    print("\n" + "=" * 60)
    print("RESULTADOS DA AVALIAÇÃO".center(60))
    print("=" * 60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision_weighted']:.4f}")
    print(f"Recall:    {metrics['recall_weighted']:.4f}")
    print(f"F1-Score:  {metrics['f1_weighted']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")

    cm = metrics["confusion_matrix"]
    print("\nConfusion Matrix:")
    print(f"  True Negative:  {cm['tn']:4d}  |  False Positive: {cm['fp']:4d}")
    print(f"  False Negative: {cm['fn']:4d}  |  True Positive:  {cm['tp']:4d}")
    print("=" * 60 + "\n")


def save_metrics(metrics: dict) -> None:
    """
    Exporta as métricas calculadas para um arquivo JSON em disco.

    Args:
        metrics (dict): Dicionário de métricas de avaliação.
    """
    output_dir = Path("metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "evaluation.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Métricas salvas em: {metrics_path}")


def log_to_mlflow(metrics: dict) -> None:
    """
    Registra as métricas de teste e o artefato JSON na última run ativa do MLflow.

    Busca a execução mais recente no experimento 'churn_prediction'.

    Args:
        metrics (dict): Dicionário de métricas de avaliação.
    """
    mlflow.set_experiment("churn_prediction")

    # Busca a última run para anexar as métricas de teste a ela
    runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=1)

    if not runs.empty:
        run_id = runs.iloc[0].run_id
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(
                {
                    "test_accuracy": metrics["accuracy"],
                    "test_precision_weighted": metrics["precision_weighted"],
                    "test_recall_weighted": metrics["recall_weighted"],
                    "test_f1_weighted": metrics["f1_weighted"],
                    "test_roc_auc": metrics["roc_auc"],
                }
            )
            mlflow.log_artifact("metrics/evaluation.json")
            print("Métricas logadas com sucesso no MLFlow.")


def main() -> None:
    """
    Orquestra o pipeline de avaliação do modelo.
    
    Inclui carregamento de recursos, cálculo de métricas, exibição 
    em console e persistência em arquivos e trackers (MLflow).
    """
    print("=" * 60)
    print("INICIANDO AVALIAÇÃO DO MODELO".center(60))
    print("=" * 60)

    # 1. Carregar modelo e dados
    model = load_model()
    X_test, y_test = load_test_data()
    print(f"Dados carregados: {X_test.shape[0]} amostras de teste\n")

    # 2. Avaliar
    metrics = evaluate_model(model, X_test, y_test)

    # 3. Resultados e Logs
    print_results(metrics)
    save_metrics(metrics)
    log_to_mlflow(metrics)

    print("Avaliação concluída com sucesso!")


if __name__ == "__main__":
    main()