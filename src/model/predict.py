from typing import Union, Tuple
import joblib
import numpy as np
import pandas as pd
from keras.models import load_model 
from pathlib import Path
import sys

def load_artifacts() -> Tuple:
    """
    Carrega o modelo Keras e o objeto preprocessor salvos em disco.

    Returns:
        tuple: (model, preprocessor) contendo o modelo neural e o transformador de colunas.
    """
    model_path = 'models/model.keras'
    preprocessor_path = 'models/preprocessor.pkl'

    print("Carregando os artefatos...")
    model = load_model(model_path)
    preprocessor = joblib.load(preprocessor_path)

    return model, preprocessor

def prepare_input_data(data: pd.DataFrame, preprocessor) -> np.ndarray:
    """
    Aplica as transformações de limpeza e encoding nos dados de entrada.

    Realiza a conversão de TotalCharges, tratamento de nulos e remoção do 
    identificador do cliente antes de aplicar o preprocessor.

    Args:
        data (pd.DataFrame): Dados brutos para predição.
        preprocessor: Objeto ColumnTransformer ajustado no treino.

    Returns:
        np.ndarray: Dados processados prontos para a rede neural.
    """
    data = data.copy()
    
    # Remover customerID se presente
    if 'customerID' in data.columns:
        data = data.drop(columns=['customerID'])
    
    # Converter TotalCharges e tratar nulos residuais
    if 'TotalCharges' in data.columns:
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())
    
    # Aplicar o pipeline de pré-processamento
    X_processed = preprocessor.transform(data)

    return X_processed

def predict(model, X: np.ndarray, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Executa a inferência do modelo para gerar classes e probabilidades.

    Args:
        model (keras.Model): Modelo treinado.
        X (np.ndarray): Matriz de features processadas.
        threshold (float): Limite de decisão para a classe positiva (Churn). 
            Defaults para 0.5.

    Returns:
        tuple[np.ndarray, np.ndarray]: (y_pred, y_pred_proba) contendo classes e scores.
    """
    # Probabilidades (0.0 a 1.0)
    y_pred_proba = model.predict(X, verbose=0).flatten()

    # Conversão para binário com base no threshold
    y_pred = (y_pred_proba >= threshold).astype(int)

    return y_pred, y_pred_proba

def predict_single(data: dict) -> dict:
    """
    Realiza a predição para uma única instância de cliente enviada como dicionário.

    Args:
        data (dict): Chaves e valores representando os dados de um cliente.

    Returns:
        dict: Resultado contendo predição binária, probabilidade e nível de risco.
    """
    # Carregar artefatos internamente
    model, preprocessor = load_artifacts()

    # Converter dicionário para DataFrame
    df = pd.DataFrame([data])

    # Preparar e predizer
    X = prepare_input_data(df, preprocessor)
    y_pred, y_pred_proba = predict(model, X)

    prob = float(y_pred_proba[0])
    
    result = {
        "churn_prediction": int(y_pred[0]),
        "churn_probability": prob,
        "churn_risk": "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
    }
    
    return result

def predict_batch(data: Union[str, pd.DataFrame], output_path: str = None) -> pd.DataFrame:
    """
    Processa múltiplas predições a partir de um arquivo CSV ou DataFrame existente.

    Args:
        data (Union[str, pd.DataFrame]): Caminho do arquivo CSV ou objeto DataFrame.
        output_path (str, optional): Caminho para salvar o CSV com resultados.

    Returns:
        pd.DataFrame: Tabela contendo IDs dos clientes (se disponíveis) e predições.
    """
    model, preprocessor = load_artifacts()

    # Carregar dados se for um caminho de arquivo
    if isinstance(data, str):
        print(f"Carregando dados de: {data}")
        data = pd.read_csv(data)
        print(f"{len(data)} registros carregados.")

    # Preservar IDs para o relatório final
    customer_ids = data['customerID'].copy() if "customerID" in data.columns else None

    # Preparar e predizer
    X = prepare_input_data(data, preprocessor)
    y_pred, y_pred_proba = predict(model, X)

    # Construir DataFrame de resultados
    results = pd.DataFrame({
        'churn_prediction': y_pred,
        'churn_probability': y_pred_proba,
        'churn_risk': ['High' if p > 0.7 else 'Medium' if p > 0.4 else 'Low' for p in y_pred_proba]
    })

    # Reanexar IDs se existirem
    if customer_ids is not None:
        results.insert(0, 'customerID', customer_ids.values)
    
    # Persistência em disco
    if output_path:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        print(f"Resultados salvos em: {output_path}")
    
    return results

def main():
    """
    Interface CLI para execução do script de predição em lote.
    """
    if len(sys.argv) < 2:
        print("Uso: python predict.py <input_path> <optional_output_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'predictions/predictions.csv'

    print(f"Processando arquivo: {input_path}")
    results = predict_batch(input_path, output_path)

    # Sumário de resultados
    print("\n" + "="*50)
    print("RESUMO DAS PREDIÇÕES".center(50))
    print("="*50)
    print(f"Total processado: {len(results)}")
    print(f"Churn detectado: {results['churn_prediction'].sum()}")
    print(f"Retenção (No Churn): {(results['churn_prediction'] == 0).sum()}")
    print("-" * 50)
    print(f"Risco ALTO:  {(results['churn_risk'] == 'High').sum()}")
    print(f"Risco MÉDIO: {(results['churn_risk'] == 'Medium').sum()}")
    print(f"Risco BAIXO: {(results['churn_risk'] == 'Low').sum()}")
    print("="*50)

if __name__ == "__main__":
    main()