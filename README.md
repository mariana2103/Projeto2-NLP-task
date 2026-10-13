# Projeto 2 — Help the doctor (classificação de especialidades médicas)

| Pasta | Conteúdo |
|---|---|
| `data/` | `train.csv` (2613 casos usáveis), `test_no_labels.csv` (teste limpo do Fenix, 398 casos), `backtranslation_cache.json` (cache da augmentation da pessoa 2) |
| `notebooks/` | `notebook1_dados_baselines` (pessoa 1), `notebook2_S1_augmentation` (pessoa 2), `notebook3_semantico_resultados` (pessoa 3) |
| `src/` | Código da pessoa 2: `parte1_base.py` (código da pessoa 1 em módulo), `s1.py` (modelo S1 + augmentation), `predict_test.py` (gera o `results.txt`) |
| `results/` | `notebook2/`: resultados de cada passo da parte 2, `s1_final_config.json` e `s1_final_model.joblib` (o modelo fica fora do git). `notebook3/`: tabela de modelos, F1 por classe, matriz de confusão, pares confundidos, candidatos a exemplos |
| `results.txt` | Previsões no teste limpo (provisório até a augmentation estar feita) |
| `relatorio/` | `main.tex` + `biblio.bib` (estrutura do template, para o Overleaf) |
| `docs/` | Enunciado, template original, `notas_parte2.md` (notas da pessoa 2) |
| `referencia/` | Projeto de 18 valores de 2024/25 (outro tema e outro template). Só para inspiração: **não vai na entrega**, fora do git |

Ordem do trabalho: notebook 1 → notebook 2 → notebook 3 → relatório.

Repositório: https://github.com/mariana2103/Projeto2-NLP-task (privado).

## Ambiente
```
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name projeto2-ln --display-name "Projeto2 LN (.venv)"
```
Nos notebooks, escolher o kernel **Projeto2 LN (.venv)**. O scikit-learn tem de ser o 1.6.1, porque o modelo S1 foi guardado com essa versão.

Atenção: o `s1_final_model.joblib` não está no git (os modelos não vão na entrega). Quem clonar o repositório tem de pedir o ficheiro à Joana ou voltar a treiná-lo com o notebook 2.

## Gerar o `results.txt`
Faz-se na secção 8 do notebook 3, ou pela linha de comandos com o script da pessoa 2. O script procura os ficheiros nas pastas dela, por isso os caminhos têm de ser dados explicitamente:
```
.venv/bin/python src/predict_test.py --test data/test_no_labels.csv --model results/notebook2/s1_final_model.joblib --out results.txt
```
Testado a 13 out: gera 398 linhas com 12 especialidades.

## Antes de submeter
O `results.txt` tem uma etiqueta por linha, sem cabeçalho, pela ordem do teste limpo. No zip não vão modelos (`*.joblib`) nem a pasta `referencia/`.
