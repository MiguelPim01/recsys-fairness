# recsys-fairness no Slurm — guia rápido de operação

Receituário do dia a dia: "quero fazer X → rode Y". Para o desenho geral
(por que 1 nó = 1 worker, retomada, libs no NFS), veja `SLURM.md`.

O ETL roda na sua máquina; os comandos Slurm abaixo rodam no **servidor**, por SSH:

```bash
ssh SEU_USUARIO@172.20.72.19
cd /mnt/cluster-nfs/datasets/$USER/recsys-fairness
```

Caminhos fixos (ajuste SEU_USUARIO/paths conforme o seu):
- código: `/mnt/cluster-nfs/datasets/$USER/recsys-fairness`
- dados processados: `.../recsys-fairness/data/processed/{lastfm,yelp}`
- libs no NFS: `/mnt/cluster-nfs/datasets/$USER/pylibs`
- resultados: `.../recsys-fairness/results/<u>_<i>/seed_<k>/`
- reservas: `slurm/reservations.txt`

---

## 0. Conceitos em uma frase

- **Seed** = uma execução completa do pipeline. A campanha é um conjunto de
  seeds (ex.: `42-46`).
- **Reserva** = você espelha a planilha do cluster em `reservations.txt`.
- **Submeter** = 1 job por máquina reservada; cada job é um *worker* que
  roda uma fatia das seeds, uma GPU por vez.
- **Encadeamento** = cada job se auto-resubmete (retomando) até terminar
  suas seeds **ou** até o FIM da reserva. Você não mexe.
- **Nada se perde**: seeds completas são puladas; seeds parciais retomam
  (amostra, splits e checkpoints já registrados são reusados).

---

## 1. Preparação (uma vez)

### 1.1 Transformar os dados na sua máquina e subir o projeto ao NFS

Na raiz do projeto local, antes de enviar qualquer experimento:

```bash
./scripts/transform_datasets.sh all
```

Acrescente `--use-restaurants-users-only` caso queira apenas usuários do Yelp
com preferência por restaurantes. Repita o ETL apenas se mudar essa
opção ou precisar reconstruir os dados processados. Espere o comando terminar
com sucesso antes de copiar os arquivos. O valor de `--restaurants-only` no
Slurm deve corresponder à opção usada aqui.

```bash
LAB_USER=SEU_USUARIO
LAB_PROJECT="/mnt/cluster-nfs/datasets/$LAB_USER/recsys-fairness"
rsync -av --exclude .git --exclude .venv --exclude results \
  --exclude data/raw --exclude data/sample \
  ./ "$LAB_USER@172.20.72.19:$LAB_PROJECT/"
```

Esse comando inclui `data/processed/` e seus manifestos. Depois do ETL,
sincronize novamente essa pasta antes de submeter novas seeds no cluster.
Não atualize os dados processados enquanto houver workers usando-os.

### 1.2 Pré-instalar as libs do projeto no NFS (NO SERVIDOR)
```bash
cd /mnt/cluster-nfs/datasets/$USER/recsys-fairness
./slurm/setup_nfs.sh
```
Isso instala recbole/numpy<2/scikit-learn/matplotlib em `pylibs/` e remove o
torch de lá (o torch bom vem do container). No fim ele importa o recbole
para confirmar que tudo casa.

### 1.3 Testar a GPU e o ambiente (2 min, partição debug)
```bash
srun --partition=debug --gres=gpu:1 --time=00:02:00 \
  apptainer exec --nv \
    -B /mnt/cluster-nfs:/mnt/cluster-nfs \
    -B /mnt/cluster-slurm:/mnt/cluster-slurm \
    /mnt/cluster-slurm/images/pytorch25-cuda128.sif \
    bash -c 'export PYTHONPATH=/mnt/cluster-nfs/datasets/$USER/recsys-fairness:/mnt/cluster-nfs/datasets/$USER/pylibs; \
      python -c "import torch,recbole; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), recbole.__version__)"'
```
Deve imprimir `True`, o nome da GPU e a versão do recbole.

---

## 2. Receitas essenciais

### Quero rodar as seeds nas máquinas que reservei
1. Confirme que `data/processed/` já foi copiado da máquina local e edite `slurm/reservations.txt` (uma linha por janela):
   ```
   DSLSERVER00    2026-10-02T00:00   2026-10-16T23:59
   DSLSERVER01    2026-10-02T00:00   2026-10-16T23:59
   ```
2. Confira o que vai ser submetido (não submete):
   ```bash
   ./slurm/submit.sh --nodelist DSLSERVER00,DSLSERVER01 --seeds 42-46 --dry-run
   ```
3. Dispare:
   ```bash
   ./slurm/submit.sh --nodelist DSLSERVER00,DSLSERVER01 --seeds 42-46
   squeue --me
   ```
As 5 seeds (42,43,44,45,46) são divididas: worker 0 pega 42,44,46; worker 1
pega 43,45.

### Quero mudar o tamanho da amostra ou os folds
```bash
./slurm/submit.sh --nodelist DSLSERVER00 --seeds 42-46 \
  --user-limit 5000 --item-limit 5000 --folds 5 --fold-workers 2
```

### Quero rodar uma seed só, sem encadear (teste)
```bash
./slurm/submit.sh --nodelist DSLSERVER00 --seeds 42 --no-chain
```

### Quero só usuários Yelp de restaurantes
```bash
./slurm/submit.sh --nodelist DSLSERVER00 --seeds 42-46 --restaurants-only
```

### Quero acompanhar o andamento
```bash
squeue --me
sacct --me -S today --format=JobID,JobName,State,Elapsed,ExitCode
tail -f rf-run-DSLSERVER00-<JOBID>.out
```

### Quero saber quantas seeds faltam (ou o status de cada uma)
```bash
# número de pendentes de um worker:
PYTHONPATH=.:pylibs_não_precisa  # (rode dentro do container; ver abaixo)
# mais simples, pela pasta de resultados — conte seeds completas:
grep -l '"status": "complete"' results/1000_1000/seed_*/experiment.json | wc -l

# tabela de status (dentro do container):
apptainer exec -B /mnt/cluster-nfs:/mnt/cluster-nfs \
  /mnt/cluster-slurm/images/pytorch25-cuda128.sif bash -c '
    cd /mnt/cluster-nfs/datasets/$USER/recsys-fairness
    export PYTHONPATH=$PWD:/mnt/cluster-nfs/datasets/$USER/pylibs
    python -m src.utils.campaign status --seeds 42-46 --worker-index 0 --worker-count 1'
```

---

## 3. Mudanças de máquina (o caso mais comum)

> Regra de ouro: **sempre que a lista de máquinas mudar, resubmeta TODAS
> juntas** com o `--nodelist` completo, para o `worker_count` ficar coerente.

### Abriu uma máquina nova
```bash
# 1) adicione a linha dela em reservations.txt;
# 2) cancele os jobs atuais:
squeue --me ; scancel <ids>
# 3) resubmeta todas juntas:
./slurm/submit.sh --nodelist DSLSERVER00,DSLSERVER01,SCISERVER00 --seeds 42-46
```
Cancelar+resubmeter é barato: retoma dos checkpoints e pula seeds completas.

### Perdi uma máquina
```bash
# ajuste/remova a linha em reservations.txt, depois:
scancel <ids>
./slurm/submit.sh --nodelist DSLSERVER00 --seeds 42-46   # só a que resta
```
As seeds da máquina perdida continuam no plano e serão pegas pelos workers
ativos.

---

## 4. Parar tudo
```bash
squeue --me
scancel <ids>        # ou: scancel -u $USER
```
`scancel` **não** dispara reencadeamento (o job entende parada manual e sai
limpo). Pode resubmeter depois; retoma de onde parou.

---

## 5. Opções do submit.sh (referência)

| Opção | Para quê |
|---|---|
| `--nodelist A,B` | máquinas reservadas (obrigatório) |
| `--seeds SPEC` | seeds: `42-46` ou `42,43,50` (obrigatório) |
| `--user-limit N` / `--item-limit N` | tamanho da amostra (default 1000) |
| `--folds N` | folds de CV (default 5) |
| `--fold-workers N` | folds em paralelo por modelo (default 1) |
| `--restaurants-only` | só usuários Yelp de restaurantes |
| `--no-chain` | desliga a auto-resubmissão |
| `--min-hours-left H` | margem de reserva (default 2) |
| `--project / --image / --pylibs / --output` | caminhos |
| `--dry-run` | mostra os `sbatch`, sem submeter |

Ajuda embutida: `./slurm/submit.sh --help`

---

## 6. Problemas comuns

### `ERRO: caminho não encontrado: .../pylibs`
Faltou rodar o `setup_nfs.sh` no servidor (passo 1.2).

### "No devices were found" / CUDA indisponível
Faltou `--gres=gpu:1` (o `.sbatch` já tem) e/ou `--nv` no apptainer (o
worker já usa). Para testar um nó: use o `srun` do passo 1.3.

### `nvidia-smi`: "Driver/library version mismatch"
O driver do nó foi atualizado sem reiniciar; a GPU só volta com reboot
(admin do nó). As outras máquinas seguem normais — rode só nelas.

### O job morreu em segundos e não reencadeou
Proteção anti-loop (< 5 min não reencadeia). Veja o `.out` para a causa
(caminho, pylibs, dados), corrija e submeta de novo.

### "seed já complete; skipping" mas eu quero refazer
Uma seed `complete` é intencionalmente pulada. Para refazer uma seed do
zero, apague a pasta dela e os dados preparados:
```bash
rm -rf results/1000_1000/seed_42 data/sample/1000_1000/seed_42
```
Depois submeta de novo.

### Mudei um script em slurm/ ou src/ na minha máquina
O servidor usa a cópia no NFS. Ressincronize com `rsync` (passo 1.1) antes
de submeter de novo. Jobs já em execução usam a versão de quando subiram.

---

## 7. Checklist para arquivar na tese (ao terminar)

- lista de seeds e `--user-limit/--item-limit` usados
- `results/<u>_<i>/seed_<k>/experiment.json` (manifestos, com os SHA-256)
- `results.json`, `grp_unfairness_and_error_table.json` e os PDFs por dataset
- imagem `.sif` usada (origem/digest)
- `pyproject.toml` / `uv.lock` (versões das libs)
