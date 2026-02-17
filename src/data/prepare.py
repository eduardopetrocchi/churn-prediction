import os
import joblib
import yaml
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


def load_params() -> dict:
    """
    Carrega as configurações de dados a partir do arquivo YAML de parâmetros.

    Returns:
        dict: Dicionário contendo as chaves de configuração sob a seção 'data'.
    """
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    return params["data"]


def load_data() -> pd.DataFrame:
    """
    Realiza a leitura do dataset bruto em formato CSV.

    Returns:
        pd.DataFrame: DataFrame contendo os dados brutos da Telco.
    """
    input_path = "data/raw/telco.csv"
    data = pd.read_csv(input_path)

    return data


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executa a limpeza inicial, tratamento de tipos e remoção de valores ausentes.

    Args:
        df (pd.DataFrame): DataFrame bruto original.

    Returns:
        pd.DataFrame: DataFrame limpo, com TotalCharges numérico e target mapeado.
    """
    df = df.copy()

    # Limpar nomes das colunas
    df.columns = df.columns.str.strip()

    # Remover customerID
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    # Converter totalCharges para numérico
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Remover NaN
    df.dropna(inplace=True)

    # Mapear variável target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide o conjunto de dados em subconjuntos de treino e teste.

    Utiliza os parâmetros 'test_size' e 'random_state' definidos no params.yaml,
    aplicando estratificação com base na coluna 'Churn'.

    Args:
        data (pd.DataFrame): DataFrame completo e limpo.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: DataFrames de treino e teste, respectivamente.
    """
    params = load_params()
    train_data, test_data = train_test_split(
        data,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=data["Churn"],
    )

    return train_data, test_data


def build_preprocessor() -> ColumnTransformer:
    """
    Instancia o objeto de pré-processamento para transformação de colunas.

    Aplica StandardScaler em colunas numéricas e OneHotEncoder em categóricas.

    Returns:
        ColumnTransformer: O pipeline de transformação configurado.
    """
    numeric_features = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )

    return preprocessor


def preprocess_data(
    train_data: pd.DataFrame, test_data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    """
    Coordena a separação de targets e a aplicação das transformações de features.

    Args:
        train_data (pd.DataFrame): Dados brutos de treino.
        test_data (pd.DataFrame): Dados brutos de teste.

    Returns:
        tuple: Contendo (X_train_df, X_test_df, y_train, y_test, preprocessor).
    """

    # Separar features (X) e target (y)
    X_train = train_data.drop("Churn", axis=1)
    y_train = train_data["Churn"]

    X_test = test_data.drop("Churn", axis=1)
    y_test = test_data["Churn"]

    # Criar e ajustar preprocessador
    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Converter arrays para Dataframes
    X_train_df = pd.DataFrame(X_train_processed)
    X_test_df = pd.DataFrame(X_test_processed)

    return X_train_df, X_test_df, y_train, y_test, preprocessor


def save_artifacts(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
) -> None:
    """
    Persiste os conjuntos de dados processados e o objeto preprocessor em disco.

    Gera arquivos CSV na pasta 'data/processed' e um arquivo PKL na pasta 'models'.

    Args:
        X_train (pd.DataFrame): Features de treino processadas.
        X_test (pd.DataFrame): Features de teste processadas.
        y_train (pd.Series): Target de treino.
        y_test (pd.Series): Target de teste.
        preprocessor (ColumnTransformer): Objeto Scikit-Learn ajustado.
    """

    data_dir = "data/processed"
    model_dir = "models"

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Salvar dados processados
    train_data = X_train.copy()
    train_data["Churn"] = y_train.values
    train_data.to_csv(os.path.join(data_dir, "train.csv"), index=False)

    test_data = X_test.copy()
    test_data["Churn"] = y_test.values
    test_data.to_csv(os.path.join(data_dir, "test.csv"), index=False)

    # Salvar preprocessador
    joblib.dump(preprocessor, os.path.join(model_dir, "preprocessor.pkl"))


def main() -> None:
    """
    Executa o pipeline completo de ETL e Pré-processamento.
    
    Orquestra desde o carregamento dos dados brutos até a exportação final
    dos artefatos prontos para o treinamento de modelos.
    """

    # 1: Carregar dados brutos
    raw_data = load_data()

    # 2: Limpar dados
    clean_data_df = clean_data(raw_data)

    # 3: Dividir em treino e teste
    train_data, test_data = split_data(clean_data_df)

    # 4: Preprocessar dados
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
        train_data, test_data
    )

    # 5: Salvar artefatos
    save_artifacts(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
    )
    print("Dados salvos com sucesso!!")
    print(f"Treino:{X_train.shape[0]} amostras, {X_train.shape[1]} features")
    print(f"Teste:{X_test.shape[0]} amostras, {X_test.shape[1]} features")


if __name__ == "__main__":
    main()