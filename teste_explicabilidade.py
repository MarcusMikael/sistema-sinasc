
from src.ml.explain_local import explicar_predicao

# Exemplo fictício para testar a integração
dados = {
    "faixa_etaria_mae": "25–29",
    "estado_civil_mae": "2.0",
    "escolaridade_mae": "4.0",
    "tipo_gravidez": "1.0",
    "qtd_gestacoes": 1,
    "qtd_partos_normais": 0,
    "qtd_partos_cesareas": 0,
    "qtd_filhos_vivos": 0,
    "qtd_filhos_mortos": 0,
    "qtd_consultas_prenatal": 6,
    "mes_inicio_prenatal": 2
}

resultado = explicar_predicao(dados)

print("\nPREDIÇÃO:")
print(resultado["predicao"])

print("\nEXPLICAÇÃO LOCAL:")
print("Score original:", resultado["explicacao_local"]["score_original"])

for item in resultado["explicacao_local"]["explicacoes"]:
    print(
        item["variavel"],
        "| Diferença:",
        item["diferenca_score"]
    )
