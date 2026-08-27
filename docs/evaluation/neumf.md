# Avaliação do NeuMF

## 1. Escopo

Protocolo inicial para validar o fluxo LastFM → RecBole → NeuMF. Prioridades:

- separar seleção de modelo e teste final;
- permitir treino simples, validação cruzada e busca independentes;
- execução determinística e inspecionável;
- poucos artefatos; resultados apenas em memória e terminal.

Não é ainda protocolo definitivo do artigo. Amostra mínima, uma época, CPU. Resultados atuais servem como teste de integração, não evidência científica.

## 2. Decisões centrais

| Tema | Decisão | Razão |
|---|---|---|
| Sinal | Feedback implícito | NeuMF canônico do RecBole otimiza presença/ausência da interação. |
| Amostra | 20 usuários × 20 itens, densa | Treino rápido e folds viáveis na prova de conceito. |
| Teste | Holdout por usuário antes da seleção | Evita escolher hiperparâmetros pelo teste. |
| Validação cruzada | K-fold por interação, estratificado por usuário | Todo usuário validado mantém histórico no treino. |
| Busca | Grade explícita e exaustiva | Transparência, reprodutibilidade, sem dependência de `hyperopt`. |
| Seleção | Maior média de NDCG@10 | Reduz dependência de um fold favorável. |
| Treino final | Todo desenvolvimento; teste uma vez | Usa os dados disponíveis após a seleção. |
| Persistência | Apenas amostra, folds e manifesto | Sem checkpoints, métricas, TensorBoard ou logs em arquivo. |

## 3. Dados e subamostra

Fonte: arquivos atômicos em `data/processed/lastfm`.

Seleção:

1. 20 itens com mais interações;
2. 20 usuários com mais interações nesses itens;
3. somente interações entre os IDs escolhidos;
4. registros correspondentes preservados em `.inter`, `.user` e `.item`.

Como o dataset processado tem uma linha por par usuário–item, frequência de interações equivale ao número de usuários por item. Empates: ID em ordem lexicográfica. Resultado atual: 20 usuários, 20 itens, 298 interações; 14–16 por usuário.

Objetivo: matriz pequena e suficientemente densa. Consequência: viés forte para popularidade e usuários muito ativos. Métricas tendem a ser mais otimistas que no LastFM completo. Não generalizar resultados desta amostra.

Os três arquivos são reutilizados quando já existem. Essa verificação é apenas por existência; mudança na fonte processada não invalida automaticamente a amostra. Para refazê-la, remover os três arquivos derivados.

## 4. Semântica do NeuMF

`play_count` é preservado e carregado, mas sua magnitude não entra na perda. Cada interação observada é um positivo. Para cada positivo de treino, um negativo uniforme é amostrado.

Modelo: NeuMF padrão do RecBole, combinação de:

- ramo MF;
- ramo MLP;
- perda binária pointwise;
- embeddings separados para MF e MLP.

Atributos de `.user` e `.item` são carregados para preservar o dataset atômico, mas o NeuMF usado não os consome. Logo, gênero, idade, país e metadados do artista não influenciam as predições nesta etapa.

Configuração-base do protótipo:

- uma época;
- batch de treino e avaliação: 64;
- learning rate: 0,001;
- embeddings MF/MLP: 64;
- MLP: `[128, 64]`;
- dropout: 0,1;
- CPU, um worker;
- seed 42; modo reproduzível e repetível.

`config/base.yaml` não é usado: foi desenhado para ratings explícitos e benchmarks diferentes deste fluxo.

## 5. Quatro modos

| CV | Busca | Seleção | Teste |
|---|---|---|---|
| Não | Não | Split aleatório 80/10/10 do RecBole | Split de teste da mesma divisão. |
| Sim | Não | Configuração-base avaliada nos K folds | Holdout final, uma vez. |
| Não | Sim | Oito candidatos avaliados apenas no fold 0 | Holdout final, uma vez. |
| Sim | Sim | Oito candidatos × K folds | Holdout final, uma vez após escolher o vencedor. |

Busca sem CV usa o fold 0 como único holdout de validação. É mais barata, porém mais sensível a uma divisão específica.

No modo simples, o RecBole faz divisão aleatória por usuário: 80% treino, 10% validação, 10% teste. Esse modo preserva o primeiro esboço; não compartilha o holdout fixo dos modos de seleção avançada.

## 6. Holdout final e folds

O RecBole 1.2.1 não oferece K-fold em `eval_args`. Decisão: pré-gerar arquivos e carregá-los por `benchmark_filename`.

### 6.1 Teste final

Para cada usuário com \(n_u\) interações:

\[
n_{teste,u} = \max(1, \lfloor 0{,}2 n_u \rfloor)
\]

Após embaralhamento determinístico, essas interações formam o teste; as demais, desenvolvimento. Na amostra atual:

- desenvolvimento: 241;
- teste: 57.

O arredondamento ocorre por usuário. Portanto, a proporção global pode diferir levemente de 20%.

### 6.2 K-fold no desenvolvimento

Padrão: 5 folds. Para cada usuário:

1. embaralhar interações com seed fixa;
2. reservar o teste;
3. distribuir desenvolvimento circularmente entre os folds;
4. em cada execução, um fold valida e os demais treinam.

Propriedades:

- cada interação de desenvolvimento valida exatamente uma vez;
- treino e validação são disjuntos em cada fold;
- todo usuário da validação aparece no treino;
- mesmos folds para todos os candidatos;
- exige ao menos K interações de desenvolvimento por usuário.

Durante seleção, o terceiro benchmark exigido pelo RecBole é um `.inter` vazio. Assim, nenhuma interação de teste é carregada ou avaliada nos folds.

### 6.3 Treino final

Após escolher a configuração:

- treino: todo desenvolvimento;
- validação: conjunto vazio;
- teste: holdout final.

Sem validação no treino final; portanto, sem early stopping. O número de épocas vem do YAML. Com uma época, “último modelo” e “melhor época” coincidem. Ao aumentar épocas, definir antes uma política de época final.

## 7. Busca de hiperparâmetros

Busca própria, não `HyperTuning` do RecBole. Motivos:

- `HyperTuning` requer `hyperopt`, ausente no ambiente;
- controle explícito do laço externo candidato → folds;
- garantia de não avaliar teste por tentativa;
- grade curta, legível e versionável.

Oito candidatos: produto de três escolhas binárias.

| Parâmetro | Valores |
|---|---|
| Learning rate | 0,0005; 0,001 |
| Dropout | 0,0; 0,1 |
| Embeddings MF e MLP | 32/32; 64/64 |

Os tamanhos MF e MLP variam juntos para limitar a busca. A arquitetura MLP `[128, 64]` permanece fixa. Busca aleatória ou bayesiana seria complexidade sem benefício para oito casos.

Para candidato \(c\), com K folds:

\[
\bar{s}_c = \frac{1}{K}\sum_{k=1}^{K}s_{c,k}
\]

\[
\sigma_c = \sqrt{\frac{1}{K}\sum_{k=1}^{K}(s_{c,k}-\bar{s}_c)^2}
\]

O vencedor maximiza \(\bar{s}_c\). Desvio-padrão populacional descreve estabilidade entre folds; não é intervalo de confiança. Empate: primeiro candidato declarado no YAML. Isso torna o desempate determinístico.

Inicialização do modelo: seed `42 + índice_do_fold`. Todos os candidatos recebem a mesma seed dentro do mesmo fold; comparação controlada. Folds sempre usam seed 42 para a divisão.

## 8. Métricas

Avaliação full-ranking do RecBole: itens candidatos são ranqueados para cada usuário; histórico conhecido é excluído da recomendação.

Métricas:

- Recall@5 e Recall@10: fração dos positivos recuperada no top-K;
- NDCG@5 e NDCG@10: acerto com desconto pela posição;
- métrica de seleção: NDCG@10;
- direção: maior é melhor.

Todas as métricas dos folds permanecem no resultado retornado pela API. Terminal mostra apenas NDCG@10 média ± desvio por candidato e métricas do teste vencedor.

## 9. Reprodutibilidade e cache

Ordem de usuários: lexicográfica. Embaralhamento: seed 42. Repetições com mesmos dados e configuração devem produzir mesmas divisões e resultados.

Folds são persistidos ao lado da amostra. Manifesto registra:

- SHA-256 do `.inter` da amostra;
- seed;
- K;
- proporção de teste;
- arquivos esperados;
- contagens.

Fold cache é reutilizado apenas se manifesto e arquivos coincidirem. Mudança no `.inter`, seed, K ou proporção regenera os folds.

O protocolo é transdutivo: `.user` e `.item` completos definem o vocabulário de usuários/itens em todos os splits. Interações de teste continuam isoladas. Adequado ao cenário de recomendar novos itens para usuários conhecidos; não mede cold start.

## 10. Separação de responsabilidades

- sampler: escolhe a subamostra e grava os três arquivos atômicos;
- splitter: cria teste, desenvolvimento, folds e manifesto;
- evaluator: treina, agrega, seleciona e testa;
- YAML do modelo: parâmetros fixos;
- YAML de busca: candidatos;
- script Python: orquestra amostra e avaliação;
- script shell: interface única de execução.

Essa separação evita lógica de amostragem dentro da avaliação e permite trocar o protocolo sem alterar a transformação original.

## 11. Saída e persistência

O logger padrão do RecBole foi silenciado: repetia configuração, treino e warnings a cada fold. Logger do projeto mostra somente:

- tamanho dos dados e número de execuções;
- candidato atual;
- progresso dos folds via `tqdm`;
- média e desvio do candidato;
- vencedor;
- métricas do teste final.

Checkpoints são desativados. TensorBoard é substituído por writer nulo. Resultados não são gravados; retornam como dicionário e aparecem resumidos no terminal. Decisão adequada ao protótipo, insuficiente para experimento final auditável. No estudo definitivo, persistir configuração resolvida, seeds, métricas por fold, ambiente e checkpoint final.

## 12. Limitações metodológicas atuais

- amostra densa e pequena; forte viés de popularidade;
- uma época: qualidade não otimizada;
- feedback implícito ignora intensidade de `play_count`;
- split aleatório por interação, não temporal;
- sem usuários ou itens novos: nenhum teste de cold start;
- sem atributos sensíveis no modelo ou métricas de fairness nesta etapa;
- folds compartilham grande parte do treino; \(\sigma\) não implica independência;
- executar o programa várias vezes permite consultar o mesmo teste várias vezes; disciplina experimental continua necessária;
- no modo simples, sem checkpoint, teste usa o estado da última época. Com várias épocas, isso pode diferir da melhor validação.

Antes de usar no artigo: ampliar dados, definir orçamento de épocas, repetir por múltiplas seeds, congelar grade e teste, persistir resultados e acrescentar métricas de fairness.

## 13. Execução

```bash
# Treino simples
./scripts/evaluate_models.sh

# Apenas validação cruzada
./scripts/evaluate_models.sh --cross-validation

# Apenas busca
./scripts/evaluate_models.sh --hyperparameter-search

# Busca com validação cruzada
./scripts/evaluate_models.sh --cross-validation --hyperparameter-search

# Outro número de folds
./scripts/evaluate_models.sh --cross-validation --folds 3
```

Execução de referência da amostra atual, seed 42, uma época, 5 folds:

- vencedor: learning rate 0,0005; dropout 0,0; embeddings 32/32;
- CV NDCG@10: 0,2916 ± 0,0416;
- teste: Recall@5 0,2833; Recall@10 0,4583; NDCG@5 0,2283; NDCG@10 0,3032.

Esses números verificam o fluxo. Não reportar como resultado final do trabalho.
