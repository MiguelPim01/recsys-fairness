# Como rodar no cluster

Guia rápido pra rodar os experimentos de fairness no cluster de GPUs do
departamento. Se é a primeira vez, lê o `GUIA-SLURM.md` do boilerplate antes
(acesso, como funciona o Slurm). Aqui eu assumo que você já tem acesso.

A ideia é simples: a campanha é um conjunto de **seeds**. Cada máquina que
você reservar vira um worker e pega uma fatia das seeds. Se um job morre, ele
volta sozinho e continua de onde parou — então você não precisa ficar
babá do terminal.

## Antes de começar

Você vai precisar de três coisas no NFS (que é compartilhado entre as
máquinas):

1. **O código** em `/mnt/cluster-nfs/datasets/$USER/recsys-fairness`
2. **Os dados processados** em `.../recsys-fairness/data/processed/{lastfm,yelp}`, copiados após o ETL local
3. **As libs do projeto** pré-instaladas (recbole etc.) — a gente faz isso no
   passo 2.

Regra de ouro do cluster: tudo que importa vive no `/mnt/cluster-nfs`, nunca
na home de um nó (que não é compartilhada).

## 1. Subir código e dados (da sua máquina)

Na raiz do projeto local, execute o ETL uma vez e espere terminar:

```bash
./scripts/transform_datasets.sh all
```

Acrescente `--use-restaurants-users-only` se quiser apenas usuários do Yelp
com preferência por restaurantes. A opção `--restaurants-only` ao
submeter os experimentos deve corresponder à variante transformada.

```bash
LAB_USER=SEU_USUARIO
LAB_PROJECT="/mnt/cluster-nfs/datasets/$LAB_USER/recsys-fairness"
rsync -av --exclude .git --exclude .venv --exclude results \
  --exclude data/raw --exclude data/sample \
  ./ "$LAB_USER@172.20.72.19:$LAB_PROJECT/"
```

Isso copia `data/processed/` e seus manifestos. Se você refizer o ETL,
sincronize os dados processados novamente antes dos próximos experimentos.
Não copie uma nova variante enquanto houver workers usando a anterior.

## 2. Instalar as libs no NFS (uma vez, no servidor)

O container tem o PyTorch, mas não tem o recbole e companhia. Como os nós de
GPU têm internet instável, a gente instala uma vez a partir do servidor:

```bash
ssh $USER@172.20.72.19
cd /mnt/cluster-nfs/datasets/$USER/recsys-fairness
./slurm/setup_nfs.sh
```

No fim ele importa o recbole pra confirmar que deu tudo certo. Se reclamar de
algo, resolve aqui antes de seguir — é mais barato que descobrir no meio de um
job.

## 3. Testar a GPU (2 minutos, partição debug)

Antes de gastar reserva, confirma que a GPU responde:

```bash
srun --partition=debug --gres=gpu:1 --time=00:02:00 \
  apptainer exec --nv \
    -B /mnt/cluster-nfs:/mnt/cluster-nfs \
    -B /mnt/cluster-slurm:/mnt/cluster-slurm \
    /mnt/cluster-slurm/images/pytorch25-cuda128.sif \
    bash -c 'export PYTHONPATH=/mnt/cluster-nfs/datasets/$USER/recsys-fairness:/mnt/cluster-nfs/datasets/$USER/pylibs; \
      python -c "import torch, recbole; print(torch.cuda.is_available(), recbole.__version__)"'
```

Tem que imprimir `True` e a versão do recbole. Se vier `No devices found`,
quase sempre é falta do `--gres=gpu:1`.

## 4. Reservar as máquinas

Edita `slurm/reservations.txt` com as máquinas que você reservou na planilha,
uma linha por janela:

```
DSLSERVER00    2026-10-02T00:00   2026-10-16T23:59
DSLSERVER01    2026-10-02T00:00   2026-10-16T23:59
```

## 5. Rodar

Primeiro um `--dry-run` pra ver o que vai ser submetido, sem enviar nada:

```bash
./slurm/submit.sh --nodelist DSLSERVER00,DSLSERVER01 --seeds 42-46 --dry-run
```

Se estiver tudo certo, tira o `--dry-run`:

```bash
./slurm/submit.sh --nodelist DSLSERVER00,DSLSERVER01 --seeds 42-46
```

As seeds 42 a 46 são divididas entre as duas máquinas. Cada job vai rodar,
morrer no limite da reserva e voltar sozinho até terminar.

Dá pra ajustar o tamanho da amostra e os folds:

```bash
./slurm/submit.sh --nodelist DSLSERVER00 --seeds 42-46 \
  --user-limit 5000 --item-limit 5000 --folds 5 --fold-workers 2
```

E pra um teste rápido de uma seed só, sem reencadear:

```bash
./slurm/submit.sh --nodelist DSLSERVER00 --seeds 42 --no-chain
```

## 6. Acompanhar

```bash
squeue --me                              # o que tá rodando (R) ou na fila (PD)
tail -f rf-run-DSLSERVER00-*.out         # log ao vivo de um worker
```

Quantas seeds já terminaram:

```bash
grep -l '"status": "complete"' \
  results/1000_1000/seed_*/experiment.json | wc -l
```

## 7. Mexer nas máquinas no meio do caminho

Regra importante: **se a lista de máquinas mudar, cancela e resubmete todas
juntas** — senão dois workers podem pegar a mesma seed.

```bash
# abri/perdi uma máquina:
# 1) ajusta reservations.txt
# 2) cancela os jobs atuais
scancel <ids>          # ou: scancel -u $USER
# 3) resubmete com a lista nova completa
./slurm/submit.sh --nodelist DSLSERVER00,DSLSERVER01,SCISERVER00 --seeds 42-46
```

Não perde nada: retoma dos checkpoints e pula o que já ficou pronto.

Cancelar na mão (`scancel`) **não** faz o job voltar — ele entende que foi
você e sai limpo. Só a morte por tempo é que reencadeia.

## Onde ficam os resultados

```
results/<user>_<item>/seed_<k>/
├── <dataset>/   # tabelas e PDFs de fairness
├── models/      # checkpoints
└── experiment.json
```

---

Precisa de mais detalhe? `OPERACAO.md` tem o receituário completo e os erros
mais comuns; `SLURM.md` explica o porquê de cada decisão de desenho.
