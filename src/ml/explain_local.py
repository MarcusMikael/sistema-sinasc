
import pandas as pd
from sklearn.model_selection import train_test_split

from src.ml.train import carregar_dados, preparar_dados
from src.ml.predict import carregar_modelo


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
