# Notas Metodológicas

Documento contendo todas as decisões metodológicas utilizadas no código.

## Dados

### LastFM-360K

- [`lastfm.py`](/home/miguel/Documents/UFES/TCC/recsys_fairness/src/data/lastfm.py)

    - Usuários que não contém nenhuma interação foram retirados | Interações sem usuário com metadados foram retiradas.
    - Artistas que não contém MBID e nem nome foram retirados.
    - Artistas sem MBID tiveram o nome encodado utilizado como ID.
    - Linhas de interações duplicadas (mesmo user_id e artist_id) foram somadas.
    - Idades não maiores do que 0 foram retiradas.

---

<center>

#### Interações

| Métrica                         |     Quantidade |
| :------------------------------ | -------------: |
| Interações totais do dataset    | **17.559.530** |
| Interações totais transformadas | **17.559.021** |

</center>

<center>

#### Limpeza e Validação

| Métrica                               | Quantidade |
| :------------------------------------ | ---------: |
| Linhas duplicadas agregadas           |    **421** |
| Artistas sem ID e nome                |      **1** |
| Valores de `play_count` não positivos |      **1** |
| Usuários que não contém metadados     |     **86** |
| Idades inválidas                      |     **57** |

</center>

<center>

#### Dimensões do Dataset

| Métrica                |  Quantidade |
| :--------------------- | ----------: |
| Quantidade de usuários | **359.347** |
| Quantidade de itens    | **268.602** |

</center>

---

### Yelp

- [`yelp.py`](/home/miguel/Documents/UFES/TCC/recsys_fairness/src/data/yelp.py)

    - Interações com comércios ou usuários sem metadados foram descartadas.
    - Usuários e comércios que possuem metadados mas não possuem interações foram descartados.

> Observações: No Yelp dataset um mesmo usuário avaliou um mesmo comércio mais de uma vez em 212.788 interações (aproximadamente 3,15%). Total de interações é 6.990.247. Por isso, na transformação dos dados foram mantidas apenas as interações mais recentes de um mesmo par usuário-item.

---

<center>

#### Interações

| Métrica                         |    Quantidade |
| :------------------------------ | ------------: |
| Interações totais do dataset    | **6.990.280** |
| Interações totais transformadas | **6.990.247** |

</center>

<center>

#### Limpeza e Validação

| Métrica                                  | Quantidade |
| :--------------------------------------- | ---------: |
| Interações removidas por usuário ausente |     **33** |

</center>

<center>

#### Dimensões do Dataset

| Métrica                |    Quantidade |
| :--------------------- | ------------: |
| Quantidade de usuários | **1.987.897** |
| Quantidade de itens    |   **150.346** |
| Usuários ativos        |   **100.370** |

</center>

<center>

#### Critérios de Atividade

| Métrica             |                   Valor |
| :------------------ | ----------------------: |
| Limiar de atividade |                  **92** |
| Data de referência  | **2022-01-19 19:48:45** |

</center>


---

## Treinamento e Avaliação

Toda a pipeline de divisão de dados, validação cruzada, busca de hiperparâmetros e resultados pode ser resumida pelo seguinte fluxograma:

```mermaid
graph LR
    A["Dados"] --> B["Dados Development"]
    A --> C["Dados Teste"]

    B --> F1["Fold 1"]
    B --> F2["Fold 2"]
    B --> F3["Fold 3"]
    B --> FN["Fold N"]

    F1 --> CV["Validação Cruzada"]
    F2 --> CV
    F3 --> CV
    FN --> CV

    CV --> T["Treino"]
    T --> TE
    C --> TE["Teste"]
    TE --> R["Resultados"]
```

Vale ressaltar alguns detalhes importantes com relação as decisões metodológicas desse fluxograma:

1. 20% dos dados são destinados para teste e 80% destinados para development.
2. O conjunto de development é dividido em um número N de folds (definido pelo usuário), em que 80% de cada fold é destinado para treino e 20% destinado para teste.
   - Todas as divisões foram feitas por usuário, ou seja, para cada usuário deixamos 20% de suas interações para teste e 80% para development. O mesmo vale para a divisão dos folds.
3. Utilizamos a validação cruzada em cima dos folds para fazer um tuning de hiperparâmetros e, após os melhores hiperparâmetros serem escolhidos treinamos o modelo com esses hiperparâmetros utilizando todo o conjunto de development (80% dos dados).
   - Para esse passo utilizamos a mediana das épocas obtidas com early stopping nos folds para utilizar ela como número máximo de épocas no treinamento final.
4. Utilizamos o melhor modelo dentre todas as épocas do passo anterior para testar a generalização dele no conjunto de teste.