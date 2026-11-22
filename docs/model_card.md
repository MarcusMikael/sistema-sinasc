# Model Card — Predição de baixo peso ao nascer

**Projeto:** Trabalho Final Integrado — Variante A (SINASC, município de São Paulo)  
**Versão do modelo:** `1.0.0`  
**Algoritmo:** `HistGradientBoostingClassifier` (scikit-learn)  
**Artefato:** `models/modelo_baixo_peso_v1.0.0.joblib`  
**Status:** modelo candidato acadêmico; não validado para uso clínico  
**Data da documentação:** 22/11/2026

## 1. Finalidade e problema

O modelo foi desenvolvido para estimar um **score preditivo de baixo peso ao nascer**, definido como peso inferior a 2.500 g, a partir de informações maternas, histórico obstétrico e dados do pré-natal presentes no SINASC.

A pergunta da **Variante A** é: **“O bebê nascerá com baixo peso (menos de 2.500 g)?”**

- **Alvo `baixo_peso = 1`:** peso ao nascer < 2.500 g.
- **Alvo `baixo_peso = 0`:** peso ao nascer ≥ 2.500 g.
- **Usuários previstos no cenário acadêmico:** equipe de saúde da família de UBS e agentes comunitários de saúde.
- **Finalidade pretendida:** apoiar estudos sobre priorização do acompanhamento pré-natal, sem substituir a avaliação de profissionais de saúde.
- **Erro considerado mais custoso:** falso negativo (um nascimento com baixo peso não sinalizado pelo modelo).
- **Métrica prioritária para a decisão operacional:** recall (sensibilidade) da classe positiva, a ser avaliado após escolha do limiar.

**Advertência:** a finalidade acima é um cenário de aplicação proposto, **não uma autorização para implantar o modelo no atendimento real**.

## 2. Dados e população de desenvolvimento

- **Origem:** arquivo `data/raw/DNRE2024.csv`, derivado dos registros do SINASC 2024 utilizados no projeto.
- **Recorte declarado do trabalho:** município de São Paulo.
- **Registros originais:** 120.786, com 37 colunas.
- **Registros válidos após remover peso ausente:** 120.785.
- **Classe positiva:** 12.472 registros (aproximadamente 10,33%).
- **Classe negativa:** 108.313 registros (aproximadamente 89,67%).
- **Separação estratificada:** 96.628 registros para treinamento (80%) e 24.157 para teste (20%), `random_state=42`.
- **Conjunto de teste:** reservado, sem uso na seleção inicial do modelo e sem métricas finais registradas nesta versão do documento.

**Limitação de representatividade:** os registros são históricos e retrospectivos; a validade para outras localidades, períodos, sistemas de coleta ou populações não foi demonstrada.

## 3. Variáveis de entrada

O modelo recebe **11 variáveis** no formato especificado em `models/input_schema.json`:

| Nome de entrada | Campo de origem no SINASC | Tipo |
|---|---|---|
| `faixa_etaria_mae` | `FxEtPartur` | Categórico |
| `estado_civil_mae` | `ESTCIVMAE` | Categórico |
| `escolaridade_mae` | `ESCMAE2010` | Categórico |
| `tipo_gravidez` | `GRAVIDEZ` | Categórico |
| `qtd_gestacoes` | `QTDGESTANT` | Numérico |
| `qtd_partos_normais` | `QTDPARTNOR` | Numérico |
| `qtd_partos_cesareas` | `QTDPARTCES` | Numérico |
| `qtd_filhos_vivos` | `QTDFILVIVO` | Numérico |
| `qtd_filhos_mortos` | `QTDFILMORT` | Numérico |
| `qtd_consultas_prenatal` | `CONSPRENAT` | Numérico |
| `mes_inicio_prenatal` | `MESPRENAT` | Numérico |

A coluna `PESO` é usada **apenas para definir o alvo**; não é entrada do modelo. O campo `RACACORMAE` foi excluído nesta versão por decisão de modelagem e consideração de equidade. As justificativas das 26 colunas excluídas estão em `docs/analise_vazamento_dados.md`.

### Atenção à disponibilidade temporal

A proposta considera a previsão **durante o acompanhamento pré-natal**, mas o SINASC registra informações após o nascimento. O total final de consultas pré-natais (`CONSPRENAT`), especialmente, pode conter informação que ainda **não estaria disponível na data de uma previsão antecipada**. A utilização prospectiva exige reconstruir as entradas conforme a informação disponível em cada momento e reavaliar o modelo. **A ausência de vazamento temporal não foi plenamente demonstrada.**

## 4. Preparação e treinamento

O script reproduzível é `src/ml/train.py`. O modelo serializado reúne o pré-processamento e o estimador em um único `Pipeline` do scikit-learn:

- **Numéricas:** conversão para número, tratamento dos códigos inválidos (`CONSPRENAT = 99` e mês fora de 1–10), imputação pela mediana com indicadores de ausência e `StandardScaler`.
- **Categóricas:** padronização de códigos ignorados, imputação constante por `"ignorado"` e `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.
- **Pré-processamento conjunto:** `ColumnTransformer`.
- **Estimador:** `HistGradientBoostingClassifier(class_weight="balanced", random_state=42)`.
- **Reprodutibilidade:** treino/teste estratificados com `random_state=42`; comparação do modelo retreinado com o artefato existente.

O balanceamento via `class_weight="balanced"` ajuda a lidar com as classes desiguais, mas altera a distribuição dos scores: **não se deve ler diretamente o score como probabilidade clínica calibrada**.

## 5. Modelos comparados e resultados

Foram comparados um baseline (`DummyClassifier`) e os modelos `HistGradientBoostingClassifier`, `RandomForestClassifier` e `LogisticRegression`, usando **validação cruzada estratificada de cinco partições apenas no treinamento**.

Resultados aproximados documentados nos experimentos:

| Configuração | PR-AUC (Average Precision) |
|---|---:|
| DummyClassifier (baseline) | 0,1033 |
| RandomForestClassifier | 0,3477 |
| LogisticRegression | 0,3467 |
| HistGradientBoostingClassifier (comparação inicial) | 0,3588 |
| HistGradientBoostingClassifier, versão selecionada sem raça/cor | **0,3593** |

Para a versão selecionada sem `RACACORMAE`, foram observados aproximadamente **recall = 0,5667** e **F2 = 0,4411** na validação cruzada, sob a configuração de classificação adotada na etapa experimental. Tais números **não são as métricas finais do conjunto de teste** e podem mudar com o limiar operacional.

**Decisão metodológica:** o recall é a principal métrica de interesse para o usuário proposto, pois o falso negativo foi considerado mais custoso. A PR-AUC foi usada como métrica relevante para comparar a capacidade de ordenação dos modelos em um cenário desbalanceado; a precisão e o F2 são complementares.

## 6. Limiar, calibração e faixas de risco

**Pendentes na versão 1.0.0.** As probabilidades fora da amostra de cada dobra (out-of-fold, OOF), produzidas no treinamento, estão em `data/processed/oof_proba.csv`, com 96.628 registros, para subsidiar o trabalho de calibração e seleção de limiar.

- O score atual **não está calibrado**.
- O limiar de decisão operacional **não foi aprovado/registrado**.
- As faixas de risco **baixo, médio e alto ainda não foram definidas**.
- O `predict_one()` atual retorna `classificacao: null`, `limiar: null` e `calibrado: false`.

Essas etapas devem ser concluídas e validadas antes da integração de classificações ou faixas de risco na interface do sistema.

## 7. Explicabilidade

### Global

A importância global foi calculada por `permutation_importance`, usando a queda na **Average Precision** em uma amostra estratificada do conjunto de treinamento, com cinco repetições. Arquivos:

- `docs/figuras/importancia_global.csv`
- `docs/figuras/importancia_global.png`

As duas maiores quedas médias observadas foram:

- `tipo_gravidez`: **0,181080**;
- `qtd_consultas_prenatal`: **0,125609**.

Esses valores **não são percentuais de influência, efeitos causais nem justificativas clínicas**.

### Local

O protótipo `src/ml/explain_local.py` compara o score de um registro com o score obtido após substituir **uma variável por vez** por uma referência estimada no conjunto de treinamento (moda nas categóricas; mediana nas numéricas).

Em um exemplo de treinamento, o score original foi `0,428166`; a alteração isolada de `qtd_filhos_vivos` produziu diferença de `+0,140588`, e a de `mes_inicio_prenatal`, `+0,060720`.

**Limitações:** essa análise é aproximada, depende da referência escolhida, pode gerar combinações pouco plausíveis e não soma contribuições exatas. O protótipo ainda precisa ser empacotado com referências persistidas e integrado à API para explicar novas previsões de forma consistente.

## 8. Reprodutibilidade e arquivos

O novo treinamento em `src/ml/train.py` reproduziu as previsões do modelo serializado para os **96.628 registros de treinamento**: diferença máxima observada **0,0**. Isso demonstra equivalência das previsões comparadas, **não** validação clínica nem avaliação independente.

| Artefato | Finalidade |
|---|---|
| `src/ml/train.py` | Preparação e reprodução do treinamento |
| `src/ml/predict.py` | Validação das entradas e geração do score |
| `src/ml/explain.py` | Importância global |
| `src/ml/explain_local.py` | Protótipo de explicação local |
| `models/modelo_baixo_peso_v1.0.0.joblib` | Pipeline serializado |
| `models/input_schema.json` | Contrato das 11 entradas |
| `models/metadata.json` | Versão, métricas e status |
| `data/processed/oof_proba.csv` | Scores OOF para calibração/limiar |
| `docs/analise_vazamento_dados.md` | Documentação das exclusões e do vazamento |
| `docs/reprodutibilidade_modelo.md` | Verificação do treinamento reproduzido |

O carregamento de artefatos `.joblib` deve ocorrer **apenas a partir de arquivos confiáveis**, devido ao risco de execução de código na desserialização.

## 9. Limitações, riscos e usos indevidos

1. **Vazamento temporal potencial:** o SINASC registra dados retrospectivos, e algumas entradas podem incluir informação acumulada após o ponto em que a equipe faria a previsão.
2. **Calibração pendente:** score bruto não equivale a probabilidade individual validada.
3. **Limiar e faixas de risco pendentes:** não há classificação operacional aprovada.
4. **Desempenho final pendente:** o conjunto de teste reservado ainda não foi avaliado na versão documentada.
5. **Generalização desconhecida:** desempenho fora do recorte e ano estudados não foi demonstrado.
6. **Equidade não comprovada:** excluir `RACACORMAE` não elimina automaticamente vieses associados a outras variáveis; faltam análises de subgrupos.
7. **Risco de alertas incorretos:** falsos negativos podem deixar casos sem sinalização, e falsos positivos podem gerar alertas adicionais e sobrecarga de acompanhamento.
8. **Explicações não causais:** importâncias e perturbações locais descrevem comportamento do modelo, não relações de causa e efeito.
9. **Qualidade dos registros:** ausência, codificação e erros de preenchimento podem alterar previsões.

**Usos inadequados:** diagnóstico clínico, decisões automáticas sobre atendimento, exclusão de pessoas de acompanhamento, comunicação de riscos individuais como certeza, substituição de profissionais ou aplicação fora da população estudada sem validação adicional.

## 10. Próximos passos e responsáveis

- **Modelagem (Integrante 1):** concluir a documentação e disponibilizar artefatos; persistir referências da explicação local e realizar verificações adicionais de consistência temporal quando possível.
- **Limiar/calibração (Integrante 2):** utilizar os scores OOF para avaliar calibração, propor limiar com atenção a recall, precisão e capacidade de atendimento, e definir faixas de risco; documentar critérios.
- **Backend (Integrante 3):** integrar o contrato de entrada e os artefatos; validar payloads; apresentar score e explicação sem alegar calibração não demonstrada.
- **Equipe:** executar avaliação final uma única vez no teste reservado após fechar as decisões de modelagem; revisar vazamento temporal, subgrupos, riscos e adequação ao uso proposto.

## 11. Histórico da versão

**v1.0.0 — candidato acadêmico:** pipeline com 11 variáveis, modelo HistGradientBoosting, artefato serializado, resultados de validação cruzada, OOF, importância global e protótipo de explicação local. Calibração, classificação, faixas de risco, avaliação final e implantação clínica não concluídas.
