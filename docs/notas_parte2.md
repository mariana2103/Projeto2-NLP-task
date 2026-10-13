# Notas da Parte 2 (pessoa 2): modelo S1 e data augmentation

## Base
- Usa os mesmos dados da pessoa 1 (2613 casos), as mesmas 5 divisões agrupadas (`StratifiedGroupKFold`, seed 42, textos repetidos sempre na mesma divisão), a mesma preparação dos textos, os mesmos pesos de classe e as mesmas métricas. O código da pessoa 1 está em `codigo/parte1_base.py`, sem alterações exceto onde procura o `train.csv`.
- Em cada passo a melhor configuração passa ao passo seguinte. A escolha usa o critério `priority()` da pessoa 1: primeiro cumprir accuracy > 0.49 e F1 mínimo ≥ 0.25, depois maior macro-F1, depois maior accuracy.
- O teste nunca foi usado para treinar nem para escolher.

## Resultados (5 divisões; valores: accuracy / macro-F1 / F1 mínimo)
| Passo | Comparação | Resultado | Escolha |
|---|---|---|---|
| 1 | Ponto de partida (melhor modelo da pessoa 1: SVM C=0.5, palavras + fragmentos nos campos curtos, q=1) | .593 / .631 / .447 | – |
| 2 | SVM C ∈ {0.1, 0.5, 1}; LR C ∈ {3, 10, 30} | SVM C=1: .597 / .633 / .458; a melhor LR (C=3) fica em .593 / .616 / .426 | **SVM, C=1** |
| 3 | Só palavras / só fragmentos / ambos / ambos + fragmentos na nota | palavras .589 / .624 / .446; fragmentos .591 / .626 / .430; ambos .597 / .633 / .458; **+ nota .596 / .636 / .463** | **palavras + fragmentos em todos os campos** |
| 4 | Pesos das classes q ∈ {0, 0.5, 1} | q=0: .595 / .563 / **.233** (falha o limite de 0.25); q=0.5: .597 / .622 / .408; **q=1: .596 / .636 / .463** | **q=1 (balanceado)** |
| 5 | Keywords strip / none / mask / raw | **strip .596 / .636 / .463**; none .595 / .632 / .438; mask .595 / .634 / .461; *raw (com a fuga da label) .922 / .908 / .839* | **strip** (raw só serve para medir a fuga) |
| 6 | Nota completa vs só secções selecionadas | **completa .596 / .636 / .463**; selecionadas .586 / .618 / .455 | **nota completa** |
| 7 | Pesos por campo (subida coordenada em {0, 0.5, 1, 2} por bloco) | nenhuma alteração melhorou o ponto de partida (.596 / .636 / .463) | **pesos mantidos**: palavras 1, fragmentos 0.5 (ver `field_weight_search.csv`) |

**Modelo final guardado** (`resultados/s1_final_model.joblib`, configuração em `resultados/s1_final_config.json`): SVM linear com C=1; palavras (uni+bigramas) e fragmentos de 3–5 caracteres em todos os campos (peso 0.5 nos fragmentos); pesos de classe q=1; 1.ª keyword retirada quando é o nome da especialidade; nota completa; treinado com os 2613 casos. Previsão no teste verificada: 398 previsões, 12 especialidades.

Principais conclusões:
- Os pesos de classe são necessários: sem eles, a classe mais fraca fica com F1 0.23, abaixo do limite.
- A fuga nas keywords vale cerca de +33 pontos de accuracy (0.596 → 0.922).
- As diferenças entre representações e classificadores são pequenas (≤ 0.01 de macro-F1).

## Por correr (passo 8)
O passo 8 está no notebook e no código, mas **ainda não tem resultados**, porque a execução foi interrompida por falta de tempo:
- **8. Augmentation:** compara `none`, `target` (classes pequenas: back-translation + recorte de secções), `all` (recorte em todas as classes + 30% sem keywords) e `copy` (controlo com cópias exatas). Demora ~10 min. O 8.1 gera os registos `augmentation_log_*.csv` para rever. Contagem já verificada na divisão 0: target 317 variantes (172 back-translation + 145 recortes), all 2368 (1848 recortes + 520 sem keywords), copy 317.

Para correr: abrir `Parte 2_S1.ipynb` e executar tudo. Os passos 1–7 são lidos do disco (o 7 volta a fazer a procura, ~25–45 min; para a saltar, usar diretamente os pesos de `field_weights_after/config.json`) (pasta `resultados/` renomeada para `Parte2_resultados/`) e não são recalculados. Se o passo 8 mudar a escolha, a secção 9 volta a treinar e guardar o modelo final.

## Como gerar o results.txt (pessoa 3)
```
pip install pandas scikit-learn nltk scipy joblib threadpoolctl
python codigo/predict_test.py --test caminho/para/test_no_labels.csv --model resultados/s1_final_model.joblib --out results.txt
```
O modelo já inclui a preparação de texto da pessoa 1, por isso recebe o ficheiro de teste original. O script confirma que há 398 linhas e que todas as etiquetas são especialidades válidas.

## Para o grupo decidir
Há duas regras que não fazem parte do S1, porque o plano pede um modelo sem fugas:
- **Regra da keyword:** usar a 1.ª keyword quando é o nome da especialidade. Cobre 320 dos 398 casos de teste.
- **Descodificação de duplicados:** 239 transcrições de teste aparecem no treino, quase sempre com outra especialidade.

Num split aleatório, estas regras levam a accuracy para ~95%. Com as divisões agrupadas da pessoa 1 não se podem avaliar. O código está em `src/decoding.py` do projeto, se o grupo as quiser usar.
