#!/bin/bash
# =====================================================================
# recsys-fairness — job de WORKER no cluster Slurm (slurm-sci-main)
#
# Modelo orientado a RESERVA (ver slurm/submit.sh e reservations.txt):
#   1 nó reservado = 1 worker = 1 job preso ao nó (-w) que se
#   AUTO-RESUBMETE até terminar suas seeds ou até o FIM da reserva.
#
# A "tarefa" é uma SEED. O worker recebe RF_WORKER_INDEX/RF_WORKER_COUNT
# via ambiente (o submit.sh injeta) e roda `src.utils.campaign run`, que
# distribui as seeds por (posição % worker_count), pula seeds completas
# e retoma seeds parciais. Cada worker escreve nas pastas de experimento
# do próprio projeto (results/<u>_<i>/seed_<k>/).
#
# NÃO submeta este arquivo direto — use slurm/submit.sh, que injeta as
# variáveis RF_* e o -w/--deadline corretos.
#
# Convenção de caminhos do cluster (idêntica no servidor e nos workers):
#   /mnt/cluster-nfs/datasets/$USER    -> entrada privada (código, dados)
#   /mnt/cluster-nfs/resultados/$USER  -> saída privada
#   /mnt/cluster-slurm/images          -> imagens .sif compartilhadas (ro)
#
# GPU: a RTX 5090 (Blackwell) exige CUDA 12.8; a imagem pytorch25-cuda128
# já traz esse stack, então roda igual nas RTX 4090 e 5090. As libs do
# projeto (recbole, numpy<2, scikit-learn, matplotlib) NÃO estão na
# imagem: são pré-instaladas no NFS (ver setup_nfs.sh) e entram via
# PYTHONPATH=$RF_PYLIBS. O torch é SEMPRE o do container (não sobrescrever).
#
# AUTO-RESUBMISSÃO: ao terminar (por concluir as seeds ou por bater no
# --deadline da reserva), o job verifica (a) ainda faltam seeds? e (b) a
# reserva ainda vale com a margem? Se ambos, resubmete a si mesmo.
# =====================================================================

#SBATCH --job-name=rf-run
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=%x-%j.out

set -euo pipefail
JOB_START="$(date +%s)"

# --- Parâmetros injetados por slurm/submit.sh ---
RF_CONTAINER="${RF_CONTAINER:-/mnt/cluster-slurm/images/pytorch25-cuda128.sif}"
RF_PROJECT="${RF_PROJECT:-/mnt/cluster-nfs/datasets/$USER/recsys-fairness}"
RF_PYLIBS="${RF_PYLIBS:-/mnt/cluster-nfs/datasets/$USER/pylibs}"
RF_SEEDS="${RF_SEEDS:?defina RF_SEEDS via submit.sh (ex.: 42-46)}"
RF_USER_LIMIT="${RF_USER_LIMIT:-1000}"
RF_ITEM_LIMIT="${RF_ITEM_LIMIT:-1000}"
RF_FOLDS="${RF_FOLDS:-5}"
RF_FOLD_WORKERS="${RF_FOLD_WORKERS:-1}"
RF_RESTAURANTS_ONLY="${RF_RESTAURANTS_ONLY:-0}"
RF_WORKER_INDEX="${RF_WORKER_INDEX:?}"
RF_WORKER_COUNT="${RF_WORKER_COUNT:?}"
RF_NODE="${RF_NODE:-}"
RF_CHAIN="${RF_CHAIN:-0}"
RF_MIN_SECONDS_LEFT="${RF_MIN_SECONDS_LEFT:-7200}"
RF_RESERVATIONS="${RF_RESERVATIONS:-$RF_PROJECT/slurm/reservations.txt}"
RF_OUTPUT="${RF_OUTPUT:-}"

RESTAURANTS_ARG=""
[[ "$RF_RESTAURANTS_ONLY" == "1" ]] && RESTAURANTS_ARG="--use-restaurants-users-only"

echo "=== recsys-fairness worker $RF_WORKER_INDEX/$RF_WORKER_COUNT ==="
echo "Job $SLURM_JOB_ID em $(hostname) — nó reservado: ${RF_NODE:-?}"
echo "GPU alocada: ${CUDA_VISIBLE_DEVICES:-nenhuma}"
echo "seeds da campanha: $RF_SEEDS | user=$RF_USER_LIMIT item=$RF_ITEM_LIMIT folds=$RF_FOLDS"
nvidia-smi -L || true

# --- Sanidade dos caminhos ---
for path in "$RF_CONTAINER" "$RF_PROJECT" "$RF_PYLIBS"; do
    if [[ ! -e "$path" ]]; then
        echo "ERRO: caminho não encontrado: $path" >&2
        [[ "$path" == "$RF_PYLIBS" ]] && echo "      Rode slurm/setup_nfs.sh no servidor para pré-instalar as libs." >&2
        exit 1
    fi
done

# Opcional: aponta results/ para uma pasta no espaço de resultados do NFS.
if [[ -n "$RF_OUTPUT" ]]; then
    mkdir -p "$RF_OUTPUT"
    if [[ ! -e "$RF_PROJECT/results" ]]; then
        ln -s "$RF_OUTPUT" "$RF_PROJECT/results"
    fi
fi

# =====================================================================
# Auto-resubmissão: chamada no fim (concluído OU morto pelo deadline).
# Roda FORA do container (precisa de sbatch).
# =====================================================================
maybe_resubmit() {
    [[ "$RF_CHAIN" == "1" ]] || { echo "encadeamento desligado (--no-chain)"; return 0; }
    [[ -n "$RF_NODE" ]] || { echo "sem RF_NODE; não reencadeia"; return 0; }

    local elapsed=$(( $(date +%s) - JOB_START ))
    if (( elapsed < 300 )); then
        echo "job durou só ${elapsed}s (<300s); provável erro persistente. NÃO reencadeia (evita loop)." >&2
        return 0
    fi

    # (a) ainda faltam seeds deste worker? (pending imprime só o número)
    local pending
    pending="$(apptainer exec \
        -B /mnt/cluster-nfs:/mnt/cluster-nfs \
        -B /mnt/cluster-slurm:/mnt/cluster-slurm \
        "$RF_CONTAINER" bash -c "
            cd '$RF_PROJECT'
            export PYTHONPATH='$RF_PROJECT:$RF_PYLIBS'
            export CUDA_VISIBLE_DEVICES=''
            python -m src.utils.campaign pending \
                --seeds '$RF_SEEDS' \
                --worker-index $RF_WORKER_INDEX \
                --worker-count $RF_WORKER_COUNT \
                --user-limit $RF_USER_LIMIT \
                --item-limit $RF_ITEM_LIMIT 2>/dev/null
        " | tail -n1)"
    if [[ ! "$pending" =~ ^[0-9]+$ ]]; then
        echo "AVISO: não apurei seeds pendentes ('$pending'); não reencadeia por segurança." >&2
        return 0
    fi
    if (( pending == 0 )); then
        echo "worker $RF_WORKER_INDEX concluiu todas as seeds atribuídas; não reencadeia."
        return 0
    fi

    # (b) reserva do nó ainda vigente com a margem?
    if ! "$RF_PROJECT/slurm/reservations.sh" valid-now "$RF_NODE" "$RF_MIN_SECONDS_LEFT" "$RF_RESERVATIONS"; then
        echo "reserva de $RF_NODE expirou/abaixo da margem; NÃO reencadeia. Faltam $pending seed(s)."
        return 0
    fi
    local deadline
    deadline="$("$RF_PROJECT/slurm/reservations.sh" deadline-iso "$RF_NODE" "$RF_RESERVATIONS")"

    echo ">> reencadeando worker $RF_WORKER_INDEX em $RF_NODE (faltam $pending, novo deadline $deadline)"
    local ev="ALL"
    ev="$ev,RF_CONTAINER=$RF_CONTAINER,RF_PROJECT=$RF_PROJECT,RF_PYLIBS=$RF_PYLIBS"
    ev="$ev,RF_SEEDS=$RF_SEEDS,RF_USER_LIMIT=$RF_USER_LIMIT,RF_ITEM_LIMIT=$RF_ITEM_LIMIT"
    ev="$ev,RF_FOLDS=$RF_FOLDS,RF_FOLD_WORKERS=$RF_FOLD_WORKERS,RF_RESTAURANTS_ONLY=$RF_RESTAURANTS_ONLY"
    ev="$ev,RF_WORKER_INDEX=$RF_WORKER_INDEX,RF_WORKER_COUNT=$RF_WORKER_COUNT,RF_NODE=$RF_NODE"
    ev="$ev,RF_CHAIN=$RF_CHAIN,RF_MIN_SECONDS_LEFT=$RF_MIN_SECONDS_LEFT,RF_RESERVATIONS=$RF_RESERVATIONS"
    [[ -n "$RF_OUTPUT" ]] && ev="$ev,RF_OUTPUT=$RF_OUTPUT"

    sbatch \
        --job-name="rf-run-$RF_NODE" \
        -w "$RF_NODE" \
        --deadline="$deadline" \
        --export="$ev" \
        "$RF_PROJECT/slurm/run-array.sbatch"
}

# --- Traps: distinguem "morte por tempo" (reencadeia) de "scancel" (não) ---
USER_CANCELLED=0; TIME_LIMIT_HIT=0
on_usr1() { TIME_LIMIT_HIT=1; echo ">> SIGUSR1 (TimeLimit/deadline): reencadeando antes de morrer."; maybe_resubmit || true; }
on_term() { USER_CANCELLED=1; echo ">> SIGTERM (scancel manual): NÃO reencadeia."; [[ -n "${APP_PID:-}" ]] && kill "$APP_PID" 2>/dev/null; exit 143; }
on_exit() { (( TIME_LIMIT_HIT )) && return 0; (( USER_CANCELLED )) && return 0; maybe_resubmit || true; }
trap on_usr1 USR1
trap on_term TERM
trap on_exit EXIT

# --- Executa o worker dentro do container (background + wait p/ os traps) ---
set +e
apptainer exec --nv \
    -B "/mnt/cluster-nfs:/mnt/cluster-nfs" \
    -B "/mnt/cluster-slurm:/mnt/cluster-slurm" \
    "$RF_CONTAINER" \
    bash -c "
        set -euo pipefail
        cd '$RF_PROJECT'
        export PYTHONUNBUFFERED=1
        export PYTHONPATH='$RF_PROJECT:$RF_PYLIBS'
        export RECSYS_PYTHON=python
        python -m src.utils.campaign run \
            --seeds '$RF_SEEDS' \
            --worker-index $RF_WORKER_INDEX \
            --worker-count $RF_WORKER_COUNT \
            --user-limit $RF_USER_LIMIT \
            --item-limit $RF_ITEM_LIMIT \
            --folds $RF_FOLDS \
            --fold-workers $RF_FOLD_WORKERS $RESTAURANTS_ARG
    " &
APP_PID=$!
wait "$APP_PID"
RUN_STATUS=$?
set -e

echo "=== worker $RF_WORKER_INDEX/$RF_WORKER_COUNT terminou (status=$RUN_STATUS) ==="
exit "$RUN_STATUS"
