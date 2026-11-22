
import pandas as pd
import json
from sklearn.model_selection import train_test_split

from src.ml.train import carregar_dados, preparar_dados
from src.ml.predict import carregar_modelo
from pathlib import Path


def calcular_referencias(X_train):
    """Calcula valores de referência usando apenas o treino."""

    referencias = {}

    categoricas = [
        "faixa_etaria_mae",
        "estado_civil_mae",
        "escolaridade_mae",
        "tipo_gravidez"
    ]

    for coluna in X_train.columns:
        if coluna in categoricas:
            referencias[coluna] = X_train[coluna].mode().iloc[0]
        else:
            referencias[coluna] = float(
                X_train[coluna].median()
            )

    return referencias


def carregar_referencias():
    """Carrega as referências já calculadas no treinamento."""

    raiz_projeto = Path(__file__).resolve().parents[2]

    caminho = (
        raiz_projeto
        / "models"
        / "referencias_explicabilidade.json"
    )

    with open(caminho, "r", encoding="utf-8") as arquivo:
        referencias = json.load(arquivo)

    return referencias



def explicar_local(modelo, entrada, referencias):
    """Mede alterações no score ao substituir uma variável."""

    if len(entrada) != 1:
        raise ValueError("A entrada deve conter apenas um registro.")

    score_original = float(
        modelo.predict_proba(entrada)[0, 1]
    )

    resultados = []

    for coluna in entrada.columns:

        entrada_alterada = entrada.copy()

        valor_original = entrada.iloc[0][coluna]
        valor_referencia = referencias[coluna]

        entrada_alterada.at[
            entrada_alterada.index[0], coluna
        ] = valor_referencia

        score_alterado = float(
            modelo.predict_proba(entrada_alterada)[0, 1]
        )

        diferenca = score_original - score_alterado

        resultados.append({
            "variavel": coluna,
            "valor_original": str(valor_original),
            "valor_referencia": str(valor_referencia),
            "diferenca_score": round(diferenca, 6)
        })

    resultados = sorted(
        resultados,
        key=lambda item: abs(item["diferenca_score"]),
        reverse=True
    )

    return {
        "score_original": round(score_original, 6),
        "explicacoes": resultados
    }


def explicar_predicao(dados):
    """
    Gera explicação local para uma entrada do backend,
    sem carregar o CSV original.
    """

    from src.ml.predict import predict_one

    # Validar a entrada com as regras já existentes
    resultado_predicao = predict_one(dados)

    # Carregar o modelo e as referências
    modelo = carregar_modelo()
    referencias = carregar_referencias()

    # Normalizar a entrada conforme o schema do modelo
    from src.ml.predict import (
        carregar_schema,
        normalizar_categoria
    )

    schema = carregar_schema()
    entrada = pd.DataFrame([dados])

    for coluna, tipo in schema["variaveis"].items():
        if tipo == "categoria":
            entrada[coluna] = (
                entrada[coluna]
                .map(normalizar_categoria)
                .astype("object")
            )
        else:
            entrada[coluna] = pd.to_numeric(
                entrada[coluna],
                errors="coerce"
            )

    # Usar a mesma ordem de colunas do schema
    entrada = entrada[list(schema["variaveis"].keys())]

    explicacao = explicar_local(
        modelo,
        entrada,
        referencias
    )

    return {
        "predicao": resultado_predicao,
        "explicacao_local": explicacao
    }



if __name__ == "__main__":

    df = carregar_dados()
    X, y = preparar_dados(df)

    X_train, _, _, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    modelo = carregar_modelo()

    referencias = calcular_referencias(X_train)


    # Salvar valores de referência para o backend
    raiz_projeto = Path(__file__).resolve().parents[2]

    caminho_referencias = (
        raiz_projeto
        / "models"
        / "referencias_explicabilidade.json"
    )

    caminho_referencias.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        caminho_referencias,
        "w",
        encoding="utf-8"
    ) as arquivo:
        json.dump(
            referencias,
            arquivo,
            ensure_ascii=False,
            indent=4,
            allow_nan=False
        )

    print("\nReferências salvas em:", caminho_referencias)


    # Exemplo utilizando um registro do treinamento
    entrada = X_train.iloc[[0]].copy()

    resultado = explicar_local(
        modelo,
        entrada,
        referencias
    )

    print("\nScore original:", resultado["score_original"])

    print("\nExplicação local:")

    for item in resultado["explicacoes"]:
        print(
            item["variavel"],
            "| Diferença:",
            item["diferenca_score"]
        )
