
import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd


RAIZ_PROJETO = Path(__file__).resolve().parents[2]
PASTA_MODELOS = RAIZ_PROJETO / "models"

CAMINHO_MODELO = (
    PASTA_MODELOS / "modelo_baixo_peso_v1.0.0.joblib"
)

CAMINHO_SCHEMA = PASTA_MODELOS / "input_schema.json"


@lru_cache(maxsize=1)
def carregar_modelo():
    return joblib.load(CAMINHO_MODELO)


@lru_cache(maxsize=1)
def carregar_schema():
    with open(CAMINHO_SCHEMA, encoding="utf-8") as arquivo:
        return json.load(arquivo)




def normalizar_categoria(valor):
    """Normaliza categorias conforme o formato do treinamento."""

    if pd.isna(valor):
        return "ignorado"

    texto = str(valor).strip()

    if texto.lower() in ["ignorado", "ign"]:
        return "ignorado"

    # Converter códigos numéricos: 1 -> "1.0"
    try:
        numero = float(texto)

        if not numero.is_integer():
            return texto

        # No dicionário SINASC, 9 representa ignorado
        if numero == 9:
            return "ignorado"

        return str(numero)

    except ValueError:
        return texto



def predict_one(dados: dict) -> dict:
    if not isinstance(dados, dict):
        raise TypeError("A entrada deve ser um dicionário.")

    schema = carregar_schema()
    variaveis = schema["variaveis"]

    faltantes = [nome for nome in variaveis if nome not in dados]
    if faltantes:
        raise ValueError(f"Campos obrigatórios ausentes: {faltantes}")

    extras = [nome for nome in dados if nome not in variaveis]
    if extras:
        raise ValueError(f"Campos desconhecidos: {extras}")

    entrada = pd.DataFrame(
        [{nome: dados[nome] for nome in variaveis}]
    )

    # Permitir a atribuição de categorias normalizadas como strings
    for nome, tipo in variaveis.items():
     if tipo == "categoria":
        entrada[nome] = entrada[nome].astype("object")

    # Validar variáveis numéricas
    for nome, regras in schema["regras_numericas"].items():
        valor = entrada.at[0, nome]

        if valor is None or pd.isna(valor):
            if not regras["permite_nulo"]:
                raise ValueError(f"{nome} não permite valor nulo.")
            entrada.at[0, nome] = float("nan")
            continue

        if isinstance(valor, bool):
            raise ValueError(f"{nome} deve ser numérico.")

        try:
            numero = float(valor)
        except (TypeError, ValueError):
            raise ValueError(f"{nome} deve ser numérico.") from None

        import math

        if not math.isfinite(numero):
            raise ValueError(f"{nome} deve ser um número finito.")

        if regras.get("inteiro") and not numero.is_integer():
            raise ValueError(f"{nome} deve ser inteiro.")

        if numero < regras["minimo"]:
            raise ValueError(
                f"{nome} deve ser maior ou igual a {regras['minimo']}."
            )

        if "maximo" in regras and numero > regras["maximo"]:
            raise ValueError(
                f"{nome} deve ser menor ou igual a {regras['maximo']}."
            )

        entrada.at[0, nome] = numero

    # Validar variáveis categóricas
    for nome, permitidas in schema["categorias_observadas"].items():
        valor = normalizar_categoria(entrada.at[0, nome])

        if valor not in permitidas:
            raise ValueError(
                f"Categoria inválida para {nome}: {valor}. "
                f"Valores aceitos: {permitidas}"
            )

        entrada.at[0, nome] = valor

    modelo = carregar_modelo()
    pontuacao = float(modelo.predict_proba(entrada)[0, 1])

    return {
        "versao_modelo": "1.0.0",
        "score_baixo_peso": round(pontuacao, 6),
        "classificacao": None,
        "limiar": None,
        "calibrado": False
    }

