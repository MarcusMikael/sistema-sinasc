
# Reprodutibilidade do Modelo de IA

## Projeto
Predição de baixo peso ao nascer — SINASC 2024.

## Objetivo
Verificar se o script `src/ml/train.py` reproduz o modelo
desenvolvido durante os experimentos no notebook.

## Base de dados
- Fonte: SINASC 2024
- Arquivo: DNRE2024.csv
- Registros originais: 120.786
- Registros válidos: 120.785
- Variáveis preditoras: 11

## Divisão dos dados
- Treinamento: 96.628 registros (80%)
- Teste: 24.157 registros (20%)
- Método: divisão estratificada
- Random state: 42

## Algoritmo
HistGradientBoostingClassifier

Configuração:
- class_weight: balanced
- random_state: 42

## Verificação de reprodutibilidade
O modelo foi treinado novamente por meio do script
`src/ml/train.py`.

As previsões foram comparadas com as do modelo
`models/modelo_baixo_peso_v1.0.0.joblib`.

Resultados:
- Registros comparados: 96.628
- Diferença máxima entre previsões: 0.0
- Resultado: previsões equivalentes

## Conclusão
O script de treinamento reproduziu as previsões
do modelo anteriormente salvo para todos os
registros da base de treinamento.

O conjunto de teste permaneceu reservado para
avaliação final, sem participação nessa comparação.

## Limitações
A equivalência das previsões não substitui a avaliação
de desempenho em dados não utilizados no treinamento.

O modelo foi desenvolvido para fins acadêmicos e de
pesquisa, não para decisões clínicas individuais.
