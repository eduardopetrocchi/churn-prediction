import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from keras.models import load_model as keras_load_model
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def load_params() -> dict:
    """
    Carrega o arquivo completo de parâmetros do projeto (params.yaml).

    Returns:
        dict: Dicionário contendo as configurações globais.
    """
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params


def load_model() -> object:
    """
    Carrega o modelo Keras pré-treinado do diretório padrão.

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


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Calcula as métricas de performance do modelo nos dados de teste.

    Gera predições de probabilidade e classe, calculando métricas como 
    F1-Score, ROC-AUC e os componentes da matriz de confusão.

    Args:
        model (keras.Model): Modelo treinado a ser avaliado.
        X_test (pd.DataFrame): Features de teste processadas.
        y_test (pd.Series): Rótulos reais de teste.

    Returns:
        tuple: (metrics_dict, y_pred, y_pred_proba)
    """
    print("Avaliando o modelo...")

    y_pred_proba = model.predict(X_test).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)

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

    return metrics, y_pred, y_pred_proba


def plot_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray, plots_dir: Path) -> None:
    """
    Gera um mapa de calor (heatmap) da matriz de confusão e o salva em disco.

    Args:
        y_test (pd.Series): Rótulos reais.
        y_pred (np.ndarray): Classes preditas pelo modelo.
        plots_dir (Path): Diretório de destino para o arquivo PNG.
    """
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    plot_path = plots_dir / "confusion_matrix.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion Matrix salva em: {plot_path}")


def plot_roc_curve(y_test: pd.Series, y_pred_proba: np.ndarray, roc_auc: float, plots_dir: Path) -> None:
    """
    Gera o gráfico da curva ROC (Receiver Operating Characteristic) e o salva em disco.

    Args:
        y_test (pd.Series): Rótulos reais.
        y_pred_proba (np.ndarray): Probabilidades preditas pelo modelo.
        roc_auc (float): Valor da área sob a curva (AUC).
        plots_dir (Path): Diretório de destino para o arquivo PNG.
    """
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    plot_path = plots_dir / "roc_curve.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROC Curve salva em: {plot_path}")


def print_results(metrics: dict) -> None:
    """
    Exibe o resumo das métricas e a matriz de confusão no console.

    Args:
        metrics (dict): Dicionário de métricas calculado em evaluate_model.
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
    Exporta o dicionário de métricas para um arquivo JSON.

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
    Registra métricas e gráficos na última execução ativa do MLflow.

    Args:
        metrics (dict): Dicionário de métricas de avaliação.
    """
    mlflow.set_experiment("churn_prediction")

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
            mlflow.log_artifact("metrics/plots/confusion_matrix.png")
            mlflow.log_artifact("metrics/plots/roc_curve.png")
            print("Métricas e gráficos logados com sucesso no MLFlow.")


def main() -> None:
    """
    Executa o pipeline completo de avaliação: carga de dados, cálculo de 
    métricas, geração de plots e persistência de logs (DVC e MLflow).
    """
    print("=" * 60)
    print("INICIANDO AVALIAÇÃO DO MODELO".center(60))
    print("=" * 60)

    # 1. Criar diretório de plots
    plots_dir = Path("metrics/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 2. Carregar modelo e dados
    model = load_model()
    X_test, y_test = load_test_data()
    print(f"Dados carregados: {X_test.shape[0]} amostras de teste\n")

    # 3. Avaliar
    metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)

    # 4. Gerar plots
    plot_confusion_matrix(y_test, y_pred, plots_dir)
    plot_roc_curve(y_test, y_pred_proba, metrics["roc_auc"], plots_dir)

    # 5. Resultados e Logs
    print_results(metrics)
    save_metrics(metrics)
    log_to_mlflow(metrics)

    print("Avaliação concluída com sucesso!")


if __name__ == "__main__":
    main()