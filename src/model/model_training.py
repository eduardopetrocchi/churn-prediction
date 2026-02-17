import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import Dense, Dropout


def build_model(input_dim: int, params: dict[str, int | float]) -> keras.Model:
    """
    Constrói e compila um modelo de rede neural feedforward para classificação binária.

    A arquitetura é baseada em camadas densas (Dense) com ativação ReLU, 
    seguidas por camadas de Dropout para regularização. A camada de saída 
    utiliza uma ativação sigmoide para predição de probabilidade.

    Args:
        input_dim (int): Número de dimensões (features) de entrada.
        params (dict[str, int | float]): Dicionário de hiperparâmetros contendo:
            - 'hidden_layers' (list ou int): Neurônios por camada oculta.
            - 'dropout' (float): Taxa de dropout (0 a 1).
            - 'learning_rate' (float): Taxa de aprendizado para o otimizador Adam.

    Returns:
        keras.Model: Modelo Keras compilado com binary_crossentropy e métrica de acurácia.
    """
    hidden_layers = params.get("hidden_layers", [64, 32])
    
    # Garantir que hidden_layers é uma lista
    if isinstance(hidden_layers, int):
        hidden_layers = [hidden_layers]
    
    # Construir camadas
    layers = []
    for i, units in enumerate(hidden_layers):
        if i == 0:
            layers.append(Dense(units=units, activation="relu", input_dim=input_dim))
        else:
            layers.append(Dense(units=units, activation="relu"))
        layers.append(Dropout(params["dropout"]))
    
    # Camada de saída para classificação binária
    layers.append(Dense(1, activation="sigmoid"))
    
    model = Sequential(layers)
    
    # Configuração do otimizador e compilação
    optimizer = keras.optimizers.Adam(learning_rate=params["learning_rate"])
    model.compile(
        optimizer=optimizer, 
        loss="binary_crossentropy", 
        metrics=["accuracy"]
    )
    
    return model