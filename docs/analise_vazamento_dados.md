# Análise de vazamento de dados e seleção de atributos — Variante A

**Projeto:** Predição de baixo peso ao nascer — SINASC, município de São Paulo  
**Versão do registro:** 1.0 (análise documental; requer validação temporal)  
**Responsável técnico:** Integrante 1 — Modelagem de IA

## 1. Problema e momento da decisão

**Pergunta preditiva:** O bebê nascerá com baixo peso (menos de 2.500 g)?  
**Usuários previstos:** equipe de saúde da família da UBS e agentes comunitários de saúde.  
**Momento pretendido:** durante o acompanhamento pré-natal, antes do nascimento.  
**Alvo:** `baixo_peso = 1` se `PESO < 2500`; caso contrário, `baixo_peso = 0`.  
**Erro prioritário:** falso negativo. **Métrica operacional prioritária:** recall da classe positiva, após a definição do limiar.

Regra metodológica: o modelo deve receber apenas informações efetivamente conhecidas no momento de cada previsão. Dados registrados somente no parto ou após o nascimento não podem ser usados como preditores de uma previsão pré-natal.

## 2. Dados e método de seleção

- Arquivo analisado: `data/raw/DNRE2024.csv`, com **120.786 linhas e 37 colunas**.
- Após descartar um registro sem `PESO` válido: **120.785 registros**.
- Variáveis preditoras selecionadas: **11**; colunas originais fora da entrada: **26**.
- **Importante:** as 26 colunas não representam 26 vazamentos comprovados. Algumas foram excluídas por desenho do modelo, redundância ou limitações de disponibilidade e governança.

## 3. Lista individual das 26 colunas não utilizadas

**Categorias de justificativa:** **VD** = vazamento direto; **PT** = informação posterior/relacionada ao parto; **MT** = momento de disponibilidade a confirmar; **SM** = seleção de modelagem; **AD** = metadado administrativo.

| Nº | Coluna | Categoria | Justificativa da não utilização |
|---:|---|---|---|
| 1 | `LOCNASC` | PT | Local efetivo em que ocorreu o nascimento, não necessariamente conhecido na consulta pré-natal. |
| 2 | `CODMUNNASC` | PT | Município efetivo do nascimento, definido pelo evento futuro. |
| 3 | `RACACORMAE` | SM | Excluída na seleção do candidato sem raça/cor materna, por consideração de equidade e comparação experimental; **não** é vazamento direto. A exclusão, isoladamente, não comprova equidade. |
| 4 | `CODUFNATU` | SM | UF de naturalidade materna; variável geográfica não selecionada nesta versão, embora possa estar disponível antes do parto. |
| 5 | `ESCMAE` | SM | Representação alternativa de escolaridade materna; priorizada `ESCMAE2010`, evitando representações redundantes. |
| 6 | `SERIESCMAE` | SM | Série escolar materna; detalhamento não selecionado além de `ESCMAE2010`. |
| 7 | `CODOCUPMAE` | SM | Código de ocupação materna; não incorporado ao esquema de entrada desta versão. Não é vazamento por natureza. |
| 8 | `CODMUNRES` | SM | Município de residência; excluído por opção de escopo e para evitar dependência geográfica do candidato. Poderia estar disponível no pré-natal. |
| 9 | `CODPAISRES` | SM | País de residência; não selecionado no esquema de entrada. |
| 10 | `SEMAGESTAC` | MT | Semanas de gestação: informação potencialmente conhecida no pré-natal, mas o valor registrado no SINASC pode refletir a duração final da gestação; não usar esse valor retrospectivo sem confirmar a equivalência temporal. |
| 11 | `TPMETESTIM` | MT | Método de estimativa da idade gestacional: o método usado no registro final pode não corresponder ao método conhecido na consulta. |
| 12 | `TPAPRESENT` | MT | Apresentação fetal do recém-nascido: pode ser conhecida em exames durante a gestação, mas o campo final da declaração não foi validado como disponível no momento escolhido. |
| 13 | `STTRABPART` | PT | Indicação de indução do trabalho de parto; depende de eventos posteriores à consulta pré-natal. |
| 14 | `PARTO` | PT | Tipo efetivo de parto (vaginal ou cesáreo); não equivale a uma intenção ou plano pré-natal. |
| 15 | `STCESPARTO` | PT | Indica se cesárea ocorreu antes do início do trabalho de parto; definido pelo evento de parto. |
| 16 | `TPNASCASSI` | PT | Quem efetivamente assistiu ao nascimento; informação do evento futuro. |
| 17 | `DTNASC2` | PT (provisório) | Pelo nome, aparenta ser campo derivado de data de nascimento; indisponível antes do nascimento. **O significado exato de `DTNASC2` precisa ser confirmado no dicionário específico do CSV**, pois o dicionário consultado apresenta `DTNASC`, não `DTNASC2`. |
| 18 | `SEXO` | MT/SM | Sexo registrado do recém-nascido; poderia estar disponível por exame pré-natal, mas a disponibilidade não foi garantida para todas as consultas e o atributo não foi selecionado. Não classificar automaticamente como vazamento. |
| 19 | `APGAR1` | PT | Avaliação de Apgar no primeiro minuto após o nascimento. |
| 20 | `APGAR5` | PT | Avaliação de Apgar no quinto minuto após o nascimento. |
| 21 | `PESO` | VD | Peso ao nascer em gramas; é usado **exclusivamente para construir o alvo**, não como entrada. Sua inclusão revelaria diretamente o resultado. |
| 22 | `IDANOMAL` | MT | Indicador de anomalia: algumas condições podem ser detectadas antes do nascimento, outras depois; o registro final não comprova disponibilidade durante o pré-natal. |
| 23 | `FaixaEtPai` | SM | Faixa etária paterna; não selecionada no esquema de entrada nesta versão. Pode ser conhecida antes do parto. |
| 24 | `RETROALIM` | AD | Status de retroalimentação do registro; metadado administrativo do SINASC, não um atributo pré-natal. |
| 25 | `TPROBSON` | MT/PT | Grupo de Robson gerado pelo sistema; é classificação obstétrica do parto, não validada como informação disponível para a previsão na consulta pré-natal. |
| 26 | `STDNNOVA` | AD | Status de declaração de nascido vivo nova; metadado do registro, não atributo clínico disponível na consulta. |

## 4. As 11 variáveis mantidas

| Coluna original | Nome usado no modelo | Condição para uso pré-natal |
|---|---|---|
| `FxEtPartur` | `faixa_etaria_mae` | Faixa etária conhecida na data da consulta. |
| `ESTCIVMAE` | `estado_civil_mae` | Situação conjugal declarada até a consulta. |
| `ESCMAE2010` | `escolaridade_mae` | Escolaridade conhecida até a consulta. |
| `GRAVIDEZ` | `tipo_gravidez` | Número de fetos/gestação única ou múltipla já identificado. |
| `QTDGESTANT` | `qtd_gestacoes` | Histórico de gestações conhecido até a consulta. |
| `QTDPARTNOR` | `qtd_partos_normais` | Partos anteriores, sem incluir o parto em previsão. |
| `QTDPARTCES` | `qtd_partos_cesareas` | Cesáreas anteriores, sem incluir o parto em previsão. |
| `QTDFILVIVO` | `qtd_filhos_vivos` | Histórico conhecido na consulta; verificar diferenças entre registro final e momento de uso. |
| `QTDFILMORT` | `qtd_filhos_mortos` | Histórico conhecido na consulta; verificar diferenças entre registro final e momento de uso. |
| `CONSPRENAT` | `qtd_consultas_prenatal` | **Ponto crítico:** usar somente consultas realizadas até a data da previsão. O total registrado ao final da gestação pode incluir consultas futuras e provocar vazamento temporal. |
| `MESPRENAT` | `mes_inicio_prenatal` | Usável apenas depois de iniciado o pré-natal; verificar consistência do registro. |

## 5. Limitação metodológica fundamental

O treino atual usa registros retrospectivos de declarações de nascidos vivos. Embora as colunas de resultado do parto tenham sido excluídas, isso **não comprova ausência de vazamento temporal** nas colunas mantidas. Em particular, `CONSPRENAT` reflete o número de consultas registrado no SINASC e pode incorporar informação que não existia na data de uma consulta anterior. Também é necessário verificar como outras variáveis são atualizadas ao longo da gestação.

Por isso, o modelo atual deve ser apresentado como **candidato acadêmico retrospectivo**, ainda **não validado para uso prospectivo clínico**. Para uso em acompanhamento pré-natal real, seria necessário definir uma data-índice por gestação, reconstruir cada entrada com informações disponíveis até essa data, treinar e avaliar novamente conforme esse desenho e realizar validação apropriada.

## 6. Procedimento implementado

No código `src/ml/train.py`, os 11 atributos foram selecionados por lista explícita. `PESO` foi utilizado para criar `baixo_peso`, mas não entrou em `X`. A divisão estratificada reservou 96.628 registros para treino e 24.157 para teste (`random_state=42`). O pipeline combina `ColumnTransformer` com imputação, padronização, one-hot encoding e `HistGradientBoostingClassifier(class_weight="balanced", random_state=42)`.

O treinamento reproduzido gerou previsões equivalentes às do modelo salvo em **96.628 amostras de treinamento** (diferença máxima observada: `0.0`). Isso demonstra consistência de reprodução, **não** prova ausência de vazamento ou validade prospectiva.

**Observação de integração:** o artefato serializado recebe os **11 campos renomeados** que compõem o esquema de entrada definido em `models/input_schema.json`. Se a disciplina exigir que a API envie diretamente os nomes originais das colunas SINASC, será necessário adicionar uma camada de mapeamento ou ajustar o pipeline; não afirmar que já recebe todos os campos brutos do CSV.

## 7. Pendências e próximos passos

1. Confirmar a definição de `DTNASC2` no dicionário específico do arquivo `DNRE2024.csv`.
2. Documentar a data exata de previsão desejada (por exemplo, uma consulta pré-natal em período específico) para verificar vazamento temporal nas 11 entradas.
3. Produzir a explicabilidade global e local e concluir o model card (requisitos 6 e 7).
4. Integrar a definição do limiar e das faixas de risco, sem interpretar o score bruto atual como probabilidade clínica calibrada.
5. Preservar o conjunto de teste para avaliação final.

## 8. Fonte documental

Prefeitura de São Paulo. **Dicionário de dados do SINASC (arquivo DBF)**. Disponível em: https://prefeitura.sp.gov.br/documents/d/saude/dicionario_de_dados_sinasc_dbf

As *descrições* dos campos presentes no dicionário foram consultadas nessa fonte; a *classificação de exclusão* e as justificativas temporais são decisões metodológicas deste projeto. Para campos derivados não documentados ali, a interpretação foi marcada como provisória.
