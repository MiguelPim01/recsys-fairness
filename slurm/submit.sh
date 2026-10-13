#!/bin/bash
# =====================================================================
# recsys-fairness — submissão no cluster Slurm (slurm-sci-main)
#
# Modelo orientado a RESERVA de máquinas (planilha do cluster espelhada
# em slurm/reservations.txt):
#
#   1 nó reservado = 1 worker = 1 job (preso ao nó com -w) que se
#   AUTO-RESUBMETE até terminar suas seeds ou até o FIM da reserva
#   daquele nó (o que vier primeiro). Cada nó tem seu próprio deadline.
#
# Por que 1 job por nó (e não um array 0-(N-1)): cada máquina pode ter
# um FIM de reserva diferente, e um array Slurm só aceita um --deadline
# comum. Submetendo um job por nó, cada cadeia respeita o prazo do SEU
# nó. O worker_count é o total de nós; cada worker recebe um índice.
#
# A "tarefa" aqui é uma SEED completa (make run_experiments SEED=k). O
# worker distribui as seeds por (posição % worker_count), roda uma de
# cada vez na GPU do nó, pula seeds já completas e retoma seeds parciais
# (reusa amostra, splits e checkpoints já registrados no manifesto).
#
# Uso:
#   ./slurm/submit.sh --nodelist NÓ1,NÓ2,... --seeds 42-46   [opções]
#   ./slurm/submit.sh --nodelist NÓ1 --seeds 42 --no-chain   (um job só)
#   ./slurm/submit.sh --nodelist NÓ1 --seeds 42-46 --dry-run (sem enviar)
#
# Opções:
#   --nodelist L         nós reservados (CSV). workers = nº de nós.
#   --seeds SPEC         seeds da campanha: inteiros e intervalos, ex.:
#                        "42-46" ou "42,43,50" (OBRIGATÓRIO).
#   --user-limit N       usuários amostrados (default 1000).
#   --item-limit N       itens amostrados (default 1000).
#   --folds N            folds de cross-validation (default 5).
#   --fold-workers N     folds em paralelo por processo de modelo (default 1).
#   --restaurants-only   só usuários Yelp com preferência por restaurantes.
#   --min-hours-left H   margem: não (re)submete se faltar < H h de reserva
#                        (default 2).
#   --reservations F     arquivo de reservas (default slurm/reservations.txt).
#   --project PATH       diretório do código recsys-fairness no NFS.
#   --output PATH        (opcional) symlink de results/ para esta pasta no NFS.
#   --image PATH         imagem .sif.
#   --pylibs PATH        libs pré-instaladas no NFS (ver setup_nfs.sh).
#   --no-chain           desliga a auto-resubmissão (1 job por nó só).
#   --dry-run            mostra os sbatch, sem submeter.
#
# Exemplos:
#   # reservei 2 máquinas; rodar seeds 42..46 divididas entre elas:
#   ./slurm/submit.sh --nodelist DSLSERVER00,DSLSERVER01 --seeds 42-46
#   # uma seed numa máquina, sem encadear (teste):
#   ./slurm/submit.sh --nodelist DSLSERVER00 --seeds 42 --no-chain
# =====================================================================

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESV="$HERE/reservations.sh"

# --- Defaults ---
NODELIST=""
SEEDS=""
USER_LIMIT=1000
ITEM_LIMIT=1000
FOLDS=5
FOLD_WORKERS=1
RESTAURANTS_ONLY=0
MIN_HOURS_LEFT=2
CHAIN=1
DRY_RUN=0
RESERVATIONS="$HERE/reservations.txt"
RF_PROJECT="${RF_PROJECT:-/mnt/cluster-nfs/datasets/$USER/recsys-fairness}"
RF_OUTPUT="${RF_OUTPUT:-}"
RF_CONTAINER="${RF_CONTAINER:-/mnt/cluster-slurm/images/pytorch25-cuda128.sif}"
RF_PYLIBS="${RF_PYLIBS:-/mnt/cluster-nfs/datasets/$USER/pylibs}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nodelist)        NODELIST="$2"; shift 2 ;;
        --seeds)           SEEDS="$2"; shift 2 ;;
        --user-limit)      USER_LIMIT="$2"; shift 2 ;;
        --item-limit)      ITEM_LIMIT="$2"; shift 2 ;;
        --folds)           FOLDS="$2"; shift 2 ;;
        --fold-workers)    FOLD_WORKERS="$2"; shift 2 ;;
        --restaurants-only) RESTAURANTS_ONLY=1; shift ;;
        --min-hours-left)  MIN_HOURS_LEFT="$2"; shift 2 ;;
        --reservations)    RESERVATIONS="$2"; shift 2 ;;
        --project)         RF_PROJECT="$2"; shift 2 ;;
        --output)          RF_OUTPUT="$2"; shift 2 ;;
        --image)           RF_CONTAINER="$2"; shift 2 ;;
        --pylibs)          RF_PYLIBS="$2"; shift 2 ;;
        --no-chain)        CHAIN=0; shift ;;
        --dry-run)         DRY_RUN=1; shift ;;
        -h|--help)         sed -n '2,60p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "opção desconhecida: $1" >&2; exit 2 ;;
    esac
done

[[ -n "$NODELIST" ]] || { echo "ERRO: informe --nodelist NÓ1,NÓ2,..." >&2; exit 2; }
[[ -n "$SEEDS" ]]    || { echo "ERRO: informe --seeds (ex.: 42-46)." >&2; exit 2; }
[[ -r "$RESERVATIONS" ]] || { echo "ERRO: reservas não encontradas: $RESERVATIONS" >&2; exit 2; }

MIN_SECONDS_LEFT=$(( MIN_HOURS_LEFT * 3600 ))

IFS=',' read -r -a NODES <<< "$NODELIST"
WORKER_COUNT="${#NODES[@]}"
(( WORKER_COUNT >= 1 )) || { echo "ERRO: --nodelist vazio" >&2; exit 2; }

echo "recsys-fairness — submissão"
echo "  project    : $RF_PROJECT"
echo "  image      : $RF_CONTAINER"
echo "  pylibs     : $RF_PYLIBS"
echo "  seeds      : $SEEDS"
echo "  limites    : user=$USER_LIMIT item=$ITEM_LIMIT folds=$FOLDS fold-workers=$FOLD_WORKERS"
echo "  restaurants-only: $([[ $RESTAURANTS_ONLY -eq 1 ]] && echo sim || echo não)"
echo "  reservas   : $RESERVATIONS (margem ${MIN_HOURS_LEFT}h)"
echo "  workers    : $WORKER_COUNT (1 por nó reservado)"
echo "  chain      : $([[ $CHAIN -eq 1 ]] && echo 'auto-resubmete até o FIM da reserva' || echo 'desligado (--no-chain)')"
echo

# Valida a reserva de TODOS os nós antes de submeter qualquer um.
for node in "${NODES[@]}"; do
    if ! "$RESV" valid-now "$node" "$MIN_SECONDS_LEFT" "$RESERVATIONS"; then
        echo "ERRO: nó '$node' não tem reserva vigente com ao menos ${MIN_HOURS_LEFT}h em $RESERVATIONS." >&2
        exit 2
    fi
done

submit() {
    if (( DRY_RUN )); then
        { printf 'DRY-RUN: sbatch --parsable'; printf ' %q' "$@"; printf '\n'; } >&2
        echo "DRY"
        return 0
    fi
    local out; out="$(sbatch --parsable "$@")"
    printf '%s' "${out%%;*}"
}

worker_index=0
for node in "${NODES[@]}"; do
    deadline="$("$RESV" deadline-iso "$node" "$RESERVATIONS")"
    echo ">> nó $node  ->  worker $worker_index/$WORKER_COUNT  (deadline $deadline)"
    ev="ALL"
    ev="$ev,RF_CONTAINER=$RF_CONTAINER,RF_PROJECT=$RF_PROJECT,RF_PYLIBS=$RF_PYLIBS"
    ev="$ev,RF_SEEDS=$SEEDS,RF_USER_LIMIT=$USER_LIMIT,RF_ITEM_LIMIT=$ITEM_LIMIT"
    ev="$ev,RF_FOLDS=$FOLDS,RF_FOLD_WORKERS=$FOLD_WORKERS,RF_RESTAURANTS_ONLY=$RESTAURANTS_ONLY"
    ev="$ev,RF_WORKER_INDEX=$worker_index,RF_WORKER_COUNT=$WORKER_COUNT,RF_NODE=$node"
    ev="$ev,RF_CHAIN=$CHAIN,RF_MIN_SECONDS_LEFT=$MIN_SECONDS_LEFT,RF_RESERVATIONS=$RESERVATIONS"
    [[ -n "$RF_OUTPUT" ]] && ev="$ev,RF_OUTPUT=$RF_OUTPUT"

    jid="$(submit \
        --job-name="rf-run-$node" \
        -w "$node" \
        --deadline="$deadline" \
        --export="$ev" \
        "$HERE/run-array.sbatch")"
    echo "   JobID=$jid"
    worker_index=$(( worker_index + 1 ))
done

echo
echo "Acompanhe:"
echo "  squeue --me"
echo "  tail -f rf-run-*-*.out"
