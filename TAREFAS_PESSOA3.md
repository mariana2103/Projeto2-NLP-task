# Pessoa 3 — o que tenho de fazer

**Entrega:** 16 out, 23:59 (Fenix). Os modelos têm de estar fechados a 13 out e o relatório pronto a 15 out.
**Notebook:** [notebooks/notebook3_semantico_resultados.ipynb](notebooks/notebook3_semantico_resultados.ipynb)

**Recebo** da pessoa 2 (Joana): os resultados da parte 1 e da parte 2, os registos de augmentation, o modelo final S1 guardado e as notas sobre as escolhas.
**Entrego** à pessoa 4 (Gonçalo): as tabelas, as figuras, os exemplos, as limitações e o `results.txt`. Eu não escrevo o relatório, explico-lhe o material.

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
- Usar **as mesmas 5 divisões da pessoa 1** (`results/notebook1/splits_seed42.json`). Os textos repetidos (mesma descrição ou mesma transcrição) ficam sempre do mesmo lado. Sem isto as comparações não valem.
- Métricas: **accuracy, macro-F1 e F1 mínima por classe**. Para os pontos automáticos é preciso acc > 49% e F1 ≥ 25% em todas as classes.
- O teste **só** serve para gerar o `results.txt`. Não entra no treino nem no vocabulário do TF-IDF.
- Guardar as previsões de validação com as colunas `row_id, gold, pred`, para a análise de erros.
- Apontar onde e como usei LLMs, porque o relatório tem de o declarar.

**Valores a bater** (melhor resultado da pessoa 1, `all_fields_balanced_class_weights`): acc 0.591, macro-F1 0.623, min-F1 0.447.

---

## 1. Comparação com o modelo semântico pré-treinado
- [ ] Opção leve, que corre no CPU: embeddings de frases (`all-MiniLM-L6-v2` ou um modelo clínico) + regressão logística.
- [ ] Opção forte, que precisa de GPU/Colab: afinar **Bio_ClinicalBERT** (`emilyalsentzer/Bio_ClinicalBERT`).
  - Entrada curta: `description + sample_name + keywords`, que cabe nos 512 tokens.
  - A transcrição tem mediana de 392 palavras, mais do que 512 tokens. Comparar truncar só o início, início+fim, e partir em pedaços com média dos logits.
- [ ] Avaliar nas mesmas 5 divisões e comparar com B1 (pessoa 1) e S1 (pessoa 2).
- [ ] Opcional (vale pontos de criatividade): um ensemble do S1 com o modelo semântico.
- [ ] Registar tudo para a tabela de hiperparâmetros: seed, versões, épocas, learning rate, batch size, max_len.

## 2. Juntar as métricas das três partes + intervalos
- [ ] Uma tabela única com baselines, B1, S1 e o modelo semântico: acc, macro-F1 e min-F1.
- [ ] Intervalos de confiança a 95% por bootstrap (a função `bootstrap_interval` já está no notebook).
- [ ] Resultados por especialidade do melhor modelo, que são obrigatórios no relatório.

## 3. Onde é que os modelos falham
- [ ] **Matriz de confusão** com nomes de classes legíveis (obrigatória no relatório, pode ser de qualquer modelo).
- [ ] F1 por especialidade: que classes caem, e se as pequenas (Dermatology 25, Psychiatry 40, Ophthalmology 71, Neurosurgery 78) melhoram com a augmentation.
- [ ] Tipos de exemplos a verificar:
  - **Surgery** contra as outras: é um tipo de procedimento, não uma especialidade (conflitos Orthopedic↔Surgery 118, Cardio↔Surgery 113, Gastro↔Surgery 95, ObGyn↔Surgery 67);
  - **Radiology ↔ Neurology / Cardio**: os mesmos relatórios de imagem;
  - **textos repetidos com etiquetas diferentes**: 701 transcrições iguais com etiquetas diferentes, o que põe um teto à accuracy;
  - campos vazios: keywords vazias em 444 casos, transcrição vazia em 33.

## 4. Exemplos de previsões certas e erradas
- [ ] Escolher **pelo menos 3 erros concretos** das nossas divisões de validação, cada um com o excerto do texto, a etiqueta prevista, a etiqueta correta e uma interpretação. A Discussão vale 3 pontos e afirmações genéricas valem 0.
- [ ] 1 ou 2 acertos que mostrem que pistas o modelo usa (comparar com as palavras de maior peso do B1).
- [ ] Rever os exemplos à mão antes de os passar ao Gonçalo.

## 5. `results.txt`
- [x] Teste limpo do Fenix em `data/test_no_labels.csv`: **398 casos** (o antigo tinha 409), sem cabeçalho, colunas `description; sample_name; transcription; keywords`.
- [ ] Usar o **modelo final da pessoa 2**, treinado com todo o treino, para prever o teste limpo.
- [ ] Gerar com `write_results`, que confirma estes pontos:
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
