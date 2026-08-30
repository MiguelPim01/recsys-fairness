# Avaliação de justiça de grupo

## Integração

A análise é executada depois da avaliação do melhor checkpoint no teste final. Ela não é
executada nos folds e seus resultados não participam da seleção de hiperparâmetros.

O ranking detalhado segue a avaliação full-sort do RecBole: o histórico do usuário é
mascarado e todas as interações do teste são consideradas relevantes. Os mesmos scores
brutos usados para o ranking produzem o NDCG global e o NDCG médio dos usuários de
cada grupo.

Para o cálculo de erro, a predição de cada interação de teste é limitada à escala `[1, 5]`.
A perda de um grupo é o MSE de todas as suas interações de teste:

\[
L_i = \frac{\sum_{(u,j) \in \Omega_{G_i}} (\hat{x}_{uj} - x_{uj})^2}
{|\Omega_{G_i}|}
\]

A injustiça entre os `g` grupos não vazios é:

\[
R_{grp} = \frac{1}{g^2}\sum_{k=1}^{g}\sum_{l>k}(L_k-L_l)^2
\]

## Agrupamentos LastFM

- gênero: masculino, feminino e desconhecido;
- idade: abaixo de 18, 18–24, 25–34, 35–44, 45–49, 50–55, acima de 55 e desconhecido;
- atividade: 5% mais ativos no development contra os demais;
- K-Means e aglomerativo: gênero, idade e atividade no development.

Para os grupos latentes, idade ausente é imputada pela mediana e acompanhada por um
indicador de ausência. A atividade recebe `log1p`, os atributos numéricos são
padronizados e gênero é representado por one-hot. K-Means usa Lloyd, `n_init=20` e a
seed do experimento; o aglomerativo usa distância euclidiana e ligação Ward.

Cada método avalia independentemente `k=2..10` pelo coeficiente de silhueta. Em caso de
empate, vence o menor `k`. Os grupos finais recebem nomes determinísticos pela ordenação
dos centroides.

## Validação por execução

Não há testes automatizados para esta feature. A validação foi feita executando o código
e auditando o JSON resultante.

Comando do fluxo simples:

```bash
./scripts/evaluate_models.sh
```

Resultados observados no teste: 50 usuários, 100 interações, NDCG@5 `0,1427` e
NDCG@10 `0,1850`. Duas execuções com seed 42 geraram o mesmo SHA-256 para o arquivo
de resultados.

Comando do fluxo com folds:

```bash
./scripts/evaluate_models.sh --cross-validation --folds 3
```

O fluxo realizou três execuções de seleção e uma única análise de justiça após o treino
final. O teste final continha 50 usuários e 225 interações, com NDCG@5 `0,1380` e
NDCG@10 `0,2128`.

Para cada agrupamento, foram verificadas as seguintes propriedades:

- soma de 50 usuários e 225 interações de teste;
- cada usuário pertence a exatamente um grupo;
- valores de `Rgrp` coincidem com o recálculo a partir dos MSEs persistidos;
- NDCG detalhado coincide com o valor agregado pelo RecBole;
- K-Means e aglomerativo registram todos os scores de silhueta e o `k` escolhido;
- o armazenamento atômico preserva entradas de outros algoritmos.

O artefato final da última execução está em `results/results_lastfm.json`.

## Gráficos e tabela do artigo

Depois de atualizar o JSON, o avaliador utiliza apenas esse arquivo para gerar os
artefatos editoriais em `results/lastfm/`:

- `boxplot.pdf`;
- `grp_unfairness_and_error_table.json`;
- `grp_unfairness_by_model_and_groups.pdf`;
- `grp_loss_by_model_and_groups.pdf`.

Os gráficos incluem atividade, idade, gênero, K-Means e aglomerativo. A tabela contém
uma linha por modelo e agrupamento, com `Rgrp` e RMSE global. Como o JSON armazena as
perdas MSE e suas quantidades de interações, o RMSE é recuperado por:

\[
RMSE = \sqrt{\frac{\sum_g MSE_g |\Omega_g|}{\sum_g |\Omega_g|}}
\]

O cálculo é repetido para todas as partições. A geração falha se elas não resultarem no
mesmo erro global, evitando publicar uma tabela baseada em dados inconsistentes.

Para regerar os arquivos sem treinar o modelo:

```bash
python -m src.utils.results --dataset lastfm
```
