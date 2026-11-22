
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

import joblib
import pandas as pd
import numpy as np

# Caminhos do projeto
RAIZ_PROJETO = Path(__file__).resolve().parents[2]

CAMINHO_DADOS = RAIZ_PROJETO / "data" / "raw" / "DNRE2024.csv"
PASTA_MODELOS = RAIZ_PROJETO / "models"

PASTA_MODELOS.mkdir(parents=True, exist_ok=True)


def carregar_dados():
    """Carrega o arquivo original do SINASC."""

    df = pd.read_csv(
        CAMINHO_DADOS,
        sep=None,
        engine="python",
        encoding="utf-8"
    )

    print("Dados carregados:", df.shape)
    print("Primeiras colunas:", df.columns[:5].tolist())

    return df


def preparar_dados(df):
    """Prepara a variável-alvo e as 11 entradas do modelo."""

    df = df.copy()

    # Converter o peso ao nascer para número
    df["PESO"] = pd.to_numeric(df["PESO"], errors="coerce")

    # Remover registros sem peso válido
    df = df.dropna(subset=["PESO"]).copy()

    # Criar a variável-alvo
    # 0 = peso igual ou superior a 2500 g
    # 1 = peso inferior a 2500 g
    df["baixo_peso"] = (df["PESO"] < 2500).astype(int)

    # Selecionar e renomear as variáveis
    colunas = {
        "FxEtPartur": "faixa_etaria_mae",
        "ESTCIVMAE": "estado_civil_mae",
        "ESCMAE2010": "escolaridade_mae",
        "GRAVIDEZ": "tipo_gravidez",
        "QTDGESTANT": "qtd_gestacoes",
        "QTDPARTNOR": "qtd_partos_normais",
        "QTDPARTCES": "qtd_partos_cesareas",
        "QTDFILVIVO": "qtd_filhos_vivos",
        "QTDFILMORT": "qtd_filhos_mortos",
        "CONSPRENAT": "qtd_consultas_prenatal",
        "MESPRENAT": "mes_inicio_prenatal"
    }

    X = df[list(colunas.keys())].rename(columns=colunas)
    y = df["baixo_peso"]

    # Variáveis numéricas
    numericas = [
        "qtd_gestacoes",
        "qtd_partos_normais",
        "qtd_partos_cesareas",
        "qtd_filhos_vivos",
        "qtd_filhos_mortos",
        "qtd_consultas_prenatal",
        "mes_inicio_prenatal"
    ]

    for coluna in numericas:
        X[coluna] = pd.to_numeric(
            X[coluna],
            errors="coerce"
        )

    # Tratar valores ignorados e inválidos
    X["qtd_consultas_prenatal"] = (
        X["qtd_consultas_prenatal"].replace(99, np.nan)
    )

    X["mes_inicio_prenatal"] = (
        X["mes_inicio_prenatal"].where(
            X["mes_inicio_prenatal"].between(1, 10),
            np.nan
        )
    )

    # Variáveis categóricas codificadas
    categoricas_codificadas = [
        "estado_civil_mae",
        "escolaridade_mae",
        "tipo_gravidez"
    ]

    for coluna in categoricas_codificadas:
        X[coluna] = pd.to_numeric(
            X[coluna],
            errors="coerce"
        )

        X[coluna] = X[coluna].replace(9, np.nan)
        X[coluna] = (
    X[coluna]
    .astype("object")
    .fillna("ignorado")
    .astype(str)
)

    # Faixa etária materna
    X["faixa_etaria_mae"] = (
        X["faixa_etaria_mae"]
        .fillna("ignorado")
        .astype(str)
        .replace("Ign", "ignorado")
    )

    return X, y


def criar_pipeline():
    """Cria o pipeline selecionado na validação cruzada."""

    colunas_numericas = [
        "qtd_gestacoes",
        "qtd_partos_normais",
        "qtd_partos_cesareas",
        "qtd_filhos_vivos",
        "qtd_filhos_mortos",
        "qtd_consultas_prenatal",
        "mes_inicio_prenatal"
    ]

    colunas_categoricas = [
        "faixa_etaria_mae",
        "estado_civil_mae",
        "escolaridade_mae",
        "tipo_gravidez"
    ]

    pipeline_numerico = Pipeline([
        (
            "imputador",
            SimpleImputer(
                strategy="median",
                add_indicator=True
            )
        ),
        ("escalonador", StandardScaler())
    ])

    pipeline_categorico = Pipeline([
        (
            "imputador",
            SimpleImputer(
                strategy="constant",
                fill_value="ignorado"
            )
        ),
        (
            "codificador",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    preprocessador = ColumnTransformer([
        (
            "numericas",
            pipeline_numerico,
            colunas_numericas
        ),
        (
            "categoricas",
            pipeline_categorico,
            colunas_categoricas
        )
    ])

    modelo = Pipeline([
        ("preprocessamento", preprocessador),
        (
            "classificador",
            HistGradientBoostingClassifier(
                class_weight="balanced",
                random_state=42
            )
        )
    ])

    return modelo



# Execução principal
if __name__ == "__main__":

    # 1. Carregar os dados
    df = carregar_dados()

    # 2. Preparar as variáveis
    X, y = preparar_dados(df)

    # 3. Exibir resultados
    print("\nDados preparados com sucesso!")

    print("Formato de X:", X.shape)
    print("Formato de y:", y.shape)

    print("\nDistribuição da variável-alvo:")
    print(y.value_counts())

    print("\nVariáveis utilizadas:")
    print(X.columns.tolist())

    print("\nExemplo das categorias:")
    print(X[
        [
            "estado_civil_mae",
            "escolaridade_mae",
            "tipo_gravidez"
        ]
    ].head())
        # 4. Separar treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nDivisão dos dados:")
    print("Treinamento:", X_train.shape)
    print("Teste:", X_test.shape)

    print("\nProporção de baixo peso:")
    print("Treinamento:", round(y_train.mean() * 100, 2), "%")
    print("Teste:", round(y_test.mean() * 100, 2), "%")

# 5. Construir o pipeline
    modelo = criar_pipeline()

    print("\nPipeline configurado com sucesso!")
    print("Etapas:", list(modelo.named_steps.keys()))

    classificador = modelo.named_steps["classificador"]
    print("Algoritmo:", type(classificador).__name__)
    print("Class weight:", classificador.class_weight)


    # 6. Treinar o modelo
    print("\nIniciando treinamento...")

    modelo.fit(X_train, y_train)

    print("Treinamento concluído com sucesso!")

    # 7. Comparar com o modelo já salvo
    caminho_modelo_anterior = (
        PASTA_MODELOS / "modelo_baixo_peso_v1.0.0.joblib"
    )

    if caminho_modelo_anterior.exists():

        modelo_anterior = joblib.load(caminho_modelo_anterior)

        # Usar as mesmas entradas nos dois modelos
        amostra = X_train

        probabilidades_novas = modelo.predict_proba(amostra)[:, 1]
        probabilidades_anteriores = (
            modelo_anterior.predict_proba(amostra)[:, 1]
        )

        diferenca_maxima = np.max(
            np.abs(probabilidades_novas - probabilidades_anteriores)
        )

        print("\nComparação com o modelo anterior:")
        print("Previsão nova:", round(probabilidades_novas[0], 6))
        print("Previsão anterior:", round(probabilidades_anteriores[0], 6))
        print("Diferença máxima:", round(float(diferenca_maxima), 8))

        if np.allclose(
            probabilidades_novas,
            probabilidades_anteriores,
            rtol=1e-6,
            atol=1e-8
        ):
            print("Resultado: previsões equivalentes!")
        else:
            print("Resultado: existem diferenças a investigar.")

    else:
        print("\nModelo anterior não encontrado.")
        print("Comparação não realizada.")

