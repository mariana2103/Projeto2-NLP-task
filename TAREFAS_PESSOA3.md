# Pessoa 3 — o que tenho de fazer

**Entrega:** 16 out, 23:59 (Fenix). Os modelos têm de estar fechados a 13 out e o relatório pronto a 15 out.
**Notebook:** [notebooks/notebook3_semantico_resultados.ipynb](notebooks/notebook3_semantico_resultados.ipynb)

**Recebi** da pessoa 2 (Joana), a 13 out: o notebook 2, o código em `src/`, os resultados em `results/notebook2/`, o modelo S1 e as notas em `docs/notas_parte2.md`. Falta a augmentation.
**Entrego** à pessoa 4 (Gonçalo): as tabelas, as figuras, os exemplos, as limitações e o `results.txt`. Eu não escrevo o relatório, explico-lhe o material.

---

## Estado final (15 out): a minha parte está feita
- [x] **Notebook 1:** verificação dos textos repetidos, os 50 casos recuperados (não mudam o B1) e as palavras de maior peso do B1. Corrido de ponta a ponta.
- [x] **Notebook 2:** passo 8 (augmentation) corrido. **Nenhuma estratégia melhora**, por isso o modelo final fica sem augmentation. Os registos das variantes estão em `results/notebook2/augmentation_log_*.csv`.
- [x] **Notebook 3:** baselines, B1, S1 e 3 modelos semânticos com intervalos de 95%, F1 por classe, matriz de confusão e análise de erros. O S1 ganha.
- [x] **`results.txt` final:** 398 linhas, sem cabeçalho, com o S1 treinado nos 2613 casos.
- [x] **Notas para o Gonçalo:** `docs/notas_pessoa3_para_relatorio.md`, com tabelas, exemplos, hiperparâmetros, limitações e o rascunho da declaração de LLMs.
- [ ] Mandar as notas ao Gonçalo e decidir com o grupo a pergunta de investigação e a regra da keyword.
- [ ] Antes de submeter: o zip sem `.joblib` e sem `referencia/`.

---

## Ponto de situação (13 out)

### Entrega da parte 2: o que recebi
- **Modelo final S1:** SVM linear com C=1; palavras e fragmentos em todos os campos; classes balanceadas (q=1); a 1.ª keyword é retirada quando é a especialidade; nota completa. Resultado: **acc .596 / macro-F1 .636 / min-F1 .463**.
  - Fica pouco acima do B1 da pessoa 1 (.593 / .631 / .447).
- **Previsão no teste:** o `src/predict_test.py` funciona. Gera 398 linhas com as 12 especialidades (comando no README).

**Problemas da parte 2 (falar com a Joana):**
1. **A augmentation (passo 8) não foi corrida.** Vale **2 pontos** e é obrigatória. Sem ela, o modelo final ainda pode mudar. É preciso decidir já quem a corre: ela ou eu.
2. **Caminhos:** o código dela procura `StudentsPack/` e `Parte2_resultados/`. O `predict_test.py` funciona com caminhos explícitos, mas o notebook 2 não corre tal como está na nossa estrutura.
3. **O `src/decoding.py`** é referido nas notas mas não veio na entrega.

**Decisão do grupo — fuga da keyword:**
- Com as keywords em bruto (a 1.ª keyword é muitas vezes o nome da especialidade), a accuracy sobe de .596 para **.922**. Isto acontece em 320 dos 398 casos de teste.
- É um campo que o próprio teste traz, por isso decidir se o `results.txt` o usa. No relatório, apresentar as duas versões.

### Notebook 3: já montado e testado (13 out)
Corre de ponta a ponta com o kernel `Projeto2 LN (.venv)`. Primeiros números, fora da divisão, nas 5 divisões da pessoa 1, com intervalo de 95%:

| Modelo | acc | macro-F1 | min-F1 |
|---|---|---|---|
| B1 (pessoa 1) | .593 [.575, .611] | .631 [.607, .652] | .447 |
| S1 (pessoa 2) | .596 [.578, .615] | .636 [.611, .657] | .463 |
| MiniLM + LR (semântico, 1.ª versão) | .540 [.520, .560] | .573 [.551, .596] | .387 |

**O que já se vê:**
- B1 e S1 são praticamente iguais: os intervalos sobrepõem-se quase por completo.
- O semântico simples fica abaixo dos dois. É um bom ponto para a pergunta de investigação, mas ainda falta testar modelos biomédicos e o início+fim da nota.
- **57%** dos casos estão num grupo com etiquetas em conflito. Nesses casos o S1 acerta só 46%; nos restantes acerta 77%.
- Os erros do S1 são quase todos **Surgery prevista como outra especialidade**: Ortho 133, Cardio 123, Gastro 113, ObGyn 77. Depois vem Radiology→Neurology (40).
- Já há um `results.txt` **provisório** na raiz (398 linhas), feito com o S1 sem augmentation.

### Parte 1: o que faltava, feito a 14 out
- **Notebook 1 corrido aqui:** os resultados estão em `results/notebook1/`. Os números mudam ligeiramente em relação à tabela que a Mariana Neto mandou (o B1 passa de .591/.623 para **.593/.631/.447**). Batem certo com o que a Joana obteve, por isso a tabela original devia vir de uma versão anterior do código.
- **Notebook 1b** (`notebooks/notebook1b_complementos_parte1.ipynb`), com o que o plano pedia e não estava feito:
  - **Verificação:** nenhum texto repetido atravessa divisões. Há 701 transcrições repetidas com mais de uma etiqueta. Campos vazios: keywords 441, transcription 30, sample_name 4, description 1.
  - **50 casos recuperados:** o pandas perde-os por causa de aspas mal fechadas, e lendo linha a linha recuperam-se. Com eles, o B1 fica igual (.593/.631/.447 → .591/.630/.453; mudam 62 de 2613 previsões). **Conclusão: não vale a pena usá-los.** Dá uma observação concreta para a Data Analysis.
  - **Palavras de maior peso do B1:** estão em `results/notebook1b/b1_top_palavras.csv`. Servem para explicar os exemplos da Discussão.

### Projeto de 18 (`referencia/`): o que lhes deu pontos
É de outro ano: tema diferente, 2 páginas e um template mais simples. O nosso template pede mais.

**O que vale a pena copiar:**
- Quatro modelos, um deles transformer (DistilBERT), com a justificação da escolha de cada um.
- Uma tabela **por classe para todos os modelos**, com o melhor de cada linha destacado. Fazer o mesmo para B1, S1 e o modelo semântico.
- A matriz de confusão do modelo escolhido.
- Na Discussão: o transformer era ligeiramente melhor nalgumas classes, mas não compensava o custo. Usar a mesma estrutura para Bio_ClinicalBERT vs S1.
- Cada exemplo tem o excerto, a linha, as etiquetas e **as palavras que levaram o modelo a errar**.

**O que temos de fazer melhor que eles:**
- **Pergunta de investigação** explícita.
- Análise de dados com pelo menos 2 observações.
- Augmentation.
- **Pelo menos 3 erros**: eles só apresentaram 1 erro e 1 acerto.
- Hiperparâmetros completos, para ser reproduzível.
- Usar macro-F1 e min-F1, não só accuracy.
- Zero gralhas.

---

## Do enunciado e do professor (o que me toca)
- **Teste limpo:** o professor publicou um teste novo, igual ao antigo mas limpo, para que as previsões de todos os grupos fiquem alinhadas. O `results.txt` tem de ser gerado **com esse teste**.
- **Treino completo:** o modelo final pode e deve ser treinado com **todo** o treino.
- **Ordem das linhas:** a linha *n* do `results.txt` corresponde à linha *n* do teste, porque a avaliação automática depende disso. Leva só etiquetas e não tem cabeçalho.
- **Pré-processamento:** o que for aplicado ao treino tem de ser aplicado da mesma forma ao teste.
- **Reprodutibilidade:** se não conseguirmos reproduzir o `results.txt`, temos **0 na avaliação automática**. O código que gera o ficheiro tem de correr do zero.
- **Etiquetas duvidosas:** é normal haver etiquetas com que não concordamos. O dataset não vai mudar, mas estes casos **devem ser discutidos** no relatório, por isso são bons exemplos para a secção 4.
- **Olhar para os dados:** "Look at the input and output!" — os exemplos concretos contam mais do que os números.
- **Instruções:** cada instrução que não for cumprida custa **-3 pontos**.
- **Oral:** pode haver discussão oral, e a nota baixa se não soubermos explicar o trabalho.

---

## Regras para todos os modelos
- Usar **as mesmas 5 divisões da pessoa 1**. O notebook 3 lê-as da coluna `fold` das previsões da pessoa 2 (`results/notebook2/*/predictions.csv`). Os textos repetidos (mesma descrição ou mesma transcrição) ficam sempre do mesmo lado. Sem isto as comparações não valem.
- Métricas: **accuracy, macro-F1 e F1 mínima por classe**. Para os pontos automáticos é preciso acc > 49% e F1 ≥ 25% em todas as classes.
- O teste **só** serve para gerar o `results.txt`. Não entra no treino nem no vocabulário do TF-IDF.
- Guardar as previsões fora da divisão no mesmo formato da pessoa 2 (`row_id, group_id, medical_specialty, prediction, fold`).
- Apontar onde e como usei LLMs, porque o relatório tem de o declarar.

**Valores a bater** (S1 final): acc .596, macro-F1 .636, min-F1 .463.

---

## 1. Comparação com o modelo semântico pré-treinado
- [x] Opção leve: embeddings `all-MiniLM-L6-v2` + regressão logística. Resultado: .540 / .573 / .387.
- [ ] Trocar `MODEL_NAME` para um modelo biomédico (`pritamdeka/S-PubMedBert-MS-MARCO`) e testar `TRUNCATION = 'head_tail'`.
- [ ] Opção forte: afinar **Bio_ClinicalBERT** (o Mac tem GPU Apple/MPS, por isso talvez corra localmente) (`emilyalsentzer/Bio_ClinicalBERT`).
  - Entrada curta: `description + sample_name + keywords`, que cabe nos 512 tokens.
  - A transcrição tem mediana de 392 palavras, mais do que 512 tokens. Comparar truncar só o início, início+fim, e partir em pedaços com média dos logits.
- [x] Avaliar nas mesmas 5 divisões e comparar com B1 e S1 (secção 4 do notebook).
- [ ] Opcional (vale pontos de criatividade): um ensemble do S1 com o modelo semântico.
- [ ] Registar tudo para a tabela de hiperparâmetros: seed, versões, épocas, learning rate, batch size, max_len.

## 2. Juntar as métricas das três partes + intervalos
- [x] Uma tabela única com B1, S1 e o modelo semântico (`results/notebook3/tabela_modelos.csv`). Falta acrescentar as baselines.
- [x] Intervalos de confiança a 95% por bootstrap.
- [x] Resultados por especialidade de todos os modelos (`f1_por_classe.csv`).

## 3. Onde é que os modelos falham
- [x] **Matriz de confusão** do S1 com nomes legíveis (`results/notebook3/confusao_S1.png`).
- [ ] F1 por especialidade: que classes caem, e se as pequenas (Dermatology 25, Psychiatry 40, Ophthalmology 71, Neurosurgery 78) melhoram com a augmentation.
- [ ] Tipos de exemplos a verificar:
  - **Surgery** contra as outras: é um tipo de procedimento, não uma especialidade (conflitos Orthopedic↔Surgery 118, Cardio↔Surgery 113, Gastro↔Surgery 95, ObGyn↔Surgery 67);
  - **Radiology ↔ Neurology / Cardio**: os mesmos relatórios de imagem;
  - **textos repetidos com etiquetas diferentes**: 701 transcrições iguais com etiquetas diferentes, o que põe um teto à accuracy;
  - campos vazios: keywords vazias em 444 casos, transcrição vazia em 33.

## 4. Exemplos de previsões certas e erradas
- [ ] Escolher, a partir de `exemplos_candidatos.csv` (com a função `show(row_id)`), **pelo menos 3 erros concretos** das nossas divisões de validação, cada um com o excerto do texto, a etiqueta prevista, a etiqueta correta e uma interpretação. A Discussão vale 3 pontos e afirmações genéricas valem 0.
- [ ] 1 ou 2 acertos que mostrem que pistas o modelo usa (comparar com as palavras de maior peso do B1).
- [ ] Rever os exemplos à mão antes de os passar ao Gonçalo.

## 5. `results.txt`
- [x] Teste limpo do Fenix em `data/test_no_labels.csv`: **398 casos** (o antigo tinha 409), sem cabeçalho, colunas `description; sample_name; transcription; keywords`.
- [x] Usar o **modelo final da pessoa 2** para prever o teste limpo. Está feito, mas é **provisório**: tem de ser repetido se a augmentation mudar o modelo.
- [x] Gerar com `write_results`, que confirma estes pontos:
  - uma etiqueta por caso: **398 linhas**;
  - sem cabeçalho;
  - pela mesma ordem do teste;
  - só etiquetas válidas.

## 6. Ponte com o Gonçalo (relatório)
Passar-lhe as tabelas e figuras organizadas e explicar-lhe:
- [ ] **Models:** o modelo semântico, de forma que dê para reproduzir.
- [ ] **Parameters/Hyperparameters:** a tabela do ponto 1.
- [ ] **Results:** tabela com os melhores modelos e intervalos, resultados por especialidade e matriz de confusão.
- [ ] **Discussion:** os exemplos do ponto 4 e a ligação à pergunta de investigação.
- [ ] **Limitações:** etiquetas em conflito, Surgery como "superclasse", classes muito pequenas, transcrições truncadas.
- [ ] **References:** Bio_ClinicalBERT / Sentence-BERT, bibliotecas e uso de LLMs (as entradas já estão em `relatorio/biblio.bib`).

## Antes de submeter (verificar com o grupo)
- [ ] O `results.txt` final foi validado.
- [ ] Outra pessoa consegue gerar o `results.txt` de novo e obtém o mesmo resultado.
- [ ] No zip vai o código e os dados aumentados, **mas não os modelos guardados**.
- [ ] Sei explicar o projeto todo, porque pode haver discussão oral.
