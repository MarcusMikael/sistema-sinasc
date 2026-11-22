
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

from src.ml.train import carregar_dados, preparar_dados


RAIZ_PROJETO = Path(__file__).resolve().parents[2]

CAMINHO_MODELO = (
    RAIZ_PROJETO
    / "models"
    / "modelo_baixo_peso_v1.0.0.joblib"
)


def calcular_importancia_global():
    """Calcula a importância global das variáveis."""

    # Carregar os dados
    df = carregar_dados()

    # Aplicar o mesmo pré-processamento do treinamento
    X, y = preparar_dados(df)

    # Reproduzir a divisão original
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Carregar o modelo treinado
    modelo = joblib.load(CAMINHO_MODELO)

    # Utilizar uma amostra estratificada do treinamento
    _, X_amostra, _, y_amostra = train_test_split(
        X_train,
        y_train,
        test_size=0.10,
        random_state=42,
        stratify=y_train
    )

    print("\nCalculando importância global...")

    # Métrica PR-AUC (Average Precision)
    def pontuacao(modelo, X_dados, y_dados):
        probabilidades = modelo.predict_proba(X_dados)[:, 1]

        return average_precision_score(
            y_dados,
            probabilidades
        )

    resultado = permutation_importance(
        modelo,
        X_amostra,
        y_amostra,
        scoring=pontuacao,
        n_repeats=5,
        random_state=42,
        n_jobs=1
    )

    importancia = pd.DataFrame({
        "variavel": X_amostra.columns,
        "importancia_media": resultado.importances_mean,
        "desvio_padrao": resultado.importances_std
    })

    importancia = importancia.sort_values(
        "importancia_media",
        ascending=False
    )

    print("\nImportância global das variáveis:")
    print(importancia.to_string(index=False))

    # Salvar resultados
    pasta_saida = RAIZ_PROJETO / "docs" / "figuras"
    pasta_saida.mkdir(parents=True, exist_ok=True)

    caminho_csv = pasta_saida / "importancia_global.csv"

    importancia.to_csv(
        caminho_csv,
        index=False
    )

    print("\nArquivo salvo em:", caminho_csv)

    
    # Gerar gráfico de importância global
    importancia_grafico = importancia.sort_values(
        "importancia_media",
        ascending=True
    )

    plt.figure(figsize=(11, 7))

    plt.barh(
        importancia_grafico["variavel"],
        importancia_grafico["importancia_media"],
        xerr=importancia_grafico["desvio_padrao"],
        capsize=3
    )

    plt.xlabel("Redução média na Average Precision")
    plt.ylabel("Variáveis")
    plt.title("Importância Global dos Atributos — SINASC 2024")

    plt.tight_layout()

    caminho_grafico = pasta_saida / "importancia_global.png"

    plt.savefig(
        caminho_grafico,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Gráfico salvo em:", caminho_grafico)



if __name__ == "__main__":
    calcular_importancia_global()


