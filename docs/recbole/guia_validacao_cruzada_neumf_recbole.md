# Guia — Validação Cruzada do NeuMF com RecBole

> **Objetivo:** usar validação cruzada para escolher configurações do **NeuMF** sem avaliar repetidamente no conjunto de teste.

---

## 1. Estratégia recomendada

O RecBole não oferece `KFold` diretamente em `eval_args`. Para validação cruzada, a abordagem mais simples é **gerar os folds previamente** e carregá-los com `benchmark_filename`.

Fluxo recomendado:

```text
Dataset original
    │
    ├── Teste final (separado uma única vez)
    │
    └── Dados de desenvolvimento
            │
            ├── Fold 0 → train / valid
            ├── Fold 1 → train / valid
            ├── Fold 2 → train / valid
            ├── ...
            └── Fold K → train / valid
```

Para cada configuração de hiperparâmetros:

1. treinar o NeuMF em todos os folds;
2. coletar a métrica de validação de cada fold;
3. calcular **média e desvio-padrão**;
4. usar a média para comparar configurações;
5. avaliar no **teste final apenas depois** de selecionar os hiperparâmetros.

---

## 2. Estrutura dos arquivos

Exemplo com 5 folds:

```text
data/sample/lastfm/
├── lastfm.fold0_train.inter
├── lastfm.fold0_valid.inter
├── lastfm.fold1_train.inter
├── lastfm.fold1_valid.inter
├── lastfm.fold2_train.inter
├── lastfm.fold2_valid.inter
├── lastfm.fold3_train.inter
├── lastfm.fold3_valid.inter
├── lastfm.fold4_train.inter
├── lastfm.fold4_valid.inter
└── lastfm.test.inter
```

O `benchmark_filename` recebe apenas os sufixos:

```python
config_dict = {
    "data_path": str(self.dataset_dir.parent.resolve()),
    "benchmark_filename": [
        f"fold{fold}_train",
        f"fold{fold}_valid",
        "test",
    ],
}
```

Assim, para `fold=0`, o RecBole carrega:

```text
lastfm.fold0_train.inter
lastfm.fold0_valid.inter
lastfm.test.inter
```

---

## 3. Adaptação do `NeuMFEvaluator`

O seu fluxo atual já contém quase tudo necessário:

```python
dataset = create_dataset(config)
train_data, valid_data, test_data = data_preparation(config, dataset)

model = get_model(config["model"])(config, train_data.dataset).to(config["device"])

trainer_class = get_trainer(config["MODEL_TYPE"], config["model"])
trainer = trainer_class(config, model)

best_valid_score, best_valid_result = trainer.fit(
    train_data,
    valid_data,
    saved=False,
)
```

A principal mudança é transformar a avaliação em algo como:

```python
def evaluate_fold(self, fold: int, hyperparams: dict | None = None):
    config_dict = {
        "data_path": str(self.dataset_dir.parent.resolve()),
        "benchmark_filename": [
            f"fold{fold}_train",
            f"fold{fold}_valid",
            "test",
        ],
        **(hyperparams or {}),
    }

    config = Config(
        model="NeuMF",
        dataset=self.dataset_dir.name,
        config_file_list=[str(self.config_path)],
        config_dict=config_dict,
    )

    init_seed(config["seed"], config["reproducibility"])

    dataset = create_dataset(config)
    train_data, valid_data, _ = data_preparation(config, dataset)

    model = get_model(config["model"])(
        config,
        train_data.dataset,
    ).to(config["device"])

    trainer_class = get_trainer(config["MODEL_TYPE"], config["model"])
    trainer = trainer_class(config, model)

    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        saved=False,
        show_progress=config["show_progress"],
    )

    return best_valid_score, best_valid_result
```

> **Importante:** durante a validação cruzada, não chame `trainer.evaluate(test_data)`.  
> O teste deve permanecer fora do processo de escolha dos hiperparâmetros.

---

## 4. Executando os folds

Uma camada acima do evaluator pode executar:

```python
import numpy as np

scores = []

for fold in range(5):
    score, _ = evaluator.evaluate_fold(fold)
    scores.append(float(score))

print(f"CV mean: {np.mean(scores):.4f}")
print(f"CV std:  {np.std(scores):.4f}")
```

O valor usado para comparar modelos deve ser:

```text
score_cv = média(score_fold_0, ..., score_fold_4)
```

Por exemplo:

```text
NDCG@10
Fold 0: 0.312
Fold 1: 0.327
Fold 2: 0.319
Fold 3: 0.330
Fold 4: 0.322

Média:  0.322
Desvio: 0.006
```

---

## 5. Validação cruzada + busca de hiperparâmetros

A busca deve ficar **por fora dos folds**:

```text
Configuração A
 ├── Fold 0
 ├── Fold 1
 ├── Fold 2
 ├── Fold 3
 └── Fold 4
      ↓
 média NDCG@10

Configuração B
 └── mesmos 5 folds
      ↓
 média NDCG@10
```

Para NeuMF, bons parâmetros para investigar são:

- `learning_rate`
- `dropout_prob`
- `mlp_hidden_size`
- `mf_embedding_size`
- `mlp_embedding_size`

A configuração vencedora é aquela com a melhor **média da métrica de validação**, e não aquela com o melhor resultado isolado em um fold.

---

## 6. Cuidados com os folds

Para NeuMF, prefira dividir as interações **por usuário**, garantindo que:

- usuários da validação também apareçam no treino;
- sempre exista histórico de treino para cada usuário avaliado;
- os mesmos folds sejam reutilizados por todas as configurações;
- o teste final nunca participe da escolha de hiperparâmetros.

Depois de escolher a melhor configuração, treine novamente usando todos os dados de desenvolvimento e execute **uma única avaliação final** no conjunto de teste.

---

## Estrutura sugerida no projeto

```text
sampler
 └── gera teste final + folds

NeuMFEvaluator
 ├── evaluate_fold(...)
 ├── executa NeuMF
 └── retorna métricas de validação

CrossValidationEvaluator
 ├── executa todos os folds
 └── calcula média/desvio

HyperparameterSearch
 ├── testa configurações
 └── seleciona maior média de validação
```

Isso permite manter o treinamento do NeuMF praticamente igual ao atual, isolando a lógica de validação cruzada em uma camada própria.

---

### Referências

- RecBole — `benchmark_filename`: https://recbole.io/docs/user_guide/config/data_settings.html
- RecBole — NeuMF: https://recbole.io/docs/user_guide/model/general/neumf.html
