#!/bin/bash
# =====================================================================
# recsys-fairness — consulta ao arquivo de reservas (slurm/reservations.txt)
#
# Fonte de verdade das janelas de alocação por máquina. Usado pelo
# submit.sh (antes de submeter) e pela auto-resubmissão do
# run-array.sbatch (antes de reencadear) para respeitar o prazo duro.
#
# Uso:
#   reservations.sh valid-now   <maquina> [min_seconds_left] [arquivo]
#       sai 0 se a máquina está reservada AGORA e a janela vigente ainda
#       tem pelo menos min_seconds_left segundos até o FIM; senão sai 1.
#       (min_seconds_left default = 0)
#
#   reservations.sh deadline-iso <maquina> [arquivo]
#       imprime o FIM (AAAA-MM-DDTHH:MM:SS) da janela que cobre AGORA.
#       Sai 1 (sem imprimir) se não houver janela vigente.
#
#   reservations.sh window-end-epoch <maquina> [arquivo]
#       igual ao deadline-iso, mas imprime o FIM em epoch (segundos).
#
# Formato do arquivo: linhas "MAQUINA INICIO FIM" com datas
# AAAA-MM-DDTHH:MM no fuso do cluster (ver reservations.txt).
# =====================================================================

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_FILE="$HERE/reservations.txt"

# Converte "AAAA-MM-DDTHH:MM[:SS]" (fuso local do cluster) para epoch.
# Ecoa vazio (e retorna 1) se a data for inválida.
_to_epoch() {
    local ts="$1"
    ts="${ts/T/ }"
    date -d "$ts" +%s 2>/dev/null
}

# Acha a janela que cobre AGORA para uma máquina e ecoa o FIM em epoch.
# Se houver várias janelas cobrindo agora (não deveria), usa a de FIM
# mais distante (mais permissiva dentro do que está reservado).
_current_window_end_epoch() {
    local machine="$1" file="$2"
    local now; now="$(date +%s)"
    local best=""
    local m start end se ee
    while read -r m start end _rest; do
        [[ -z "${m:-}" || "${m:0:1}" == "#" ]] && continue
        [[ -z "${start:-}" || -z "${end:-}" ]] && continue
        [[ "$m" != "$machine" ]] && continue
        se="$(_to_epoch "$start")" || continue
        ee="$(_to_epoch "$end")" || continue
        [[ -z "$se" || -z "$ee" ]] && continue
        if (( now >= se && now <= ee )); then
            if [[ -z "$best" ]] || (( ee > best )); then
                best="$ee"
            fi
        fi
    done < "$file"
    [[ -n "$best" ]] && printf '%s' "$best"
}

cmd="${1:-}"; shift || true

case "$cmd" in
    valid-now)
        machine="${1:?uso: valid-now <maquina> [min_seconds_left] [arquivo]}"
        min_left="${2:-0}"
        file="${3:-$DEFAULT_FILE}"
        [[ -r "$file" ]] || { echo "ERRO: arquivo de reservas não encontrado: $file" >&2; exit 2; }
        end="$(_current_window_end_epoch "$machine" "$file")"
        [[ -z "$end" ]] && exit 1
        now="$(date +%s)"
        (( end - now >= min_left )) && exit 0 || exit 1
        ;;
    deadline-iso)
        machine="${1:?uso: deadline-iso <maquina> [arquivo]}"
        file="${2:-$DEFAULT_FILE}"
        [[ -r "$file" ]] || { echo "ERRO: arquivo de reservas não encontrado: $file" >&2; exit 2; }
        end="$(_current_window_end_epoch "$machine" "$file")"
        [[ -z "$end" ]] && exit 1
        date -d "@$end" +%Y-%m-%dT%H:%M:%S
        ;;
    window-end-epoch)
        machine="${1:?uso: window-end-epoch <maquina> [arquivo]}"
        file="${2:-$DEFAULT_FILE}"
        [[ -r "$file" ]] || { echo "ERRO: arquivo de reservas não encontrado: $file" >&2; exit 2; }
        end="$(_current_window_end_epoch "$machine" "$file")"
        [[ -z "$end" ]] && exit 1
        printf '%s' "$end"
        ;;
    ""|-h|--help)
        sed -n '2,33p' "${BASH_SOURCE[0]}"
        ;;
    *)
        echo "subcomando desconhecido: $cmd" >&2
        exit 2
        ;;
esac
