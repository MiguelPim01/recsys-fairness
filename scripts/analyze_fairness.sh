#!/usr/bin/env bash

set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_root"

model="all"
dataset="all"
user_limit=1000
item_limit=1000
experiment=""

show_help() {
    cat <<'EOF'
Usage: scripts/analyze_fairness.sh [options]

Regenerate fairness artifacts from saved model checkpoints without training.

Options:
  --model MODEL          Model to analyze: neumf, multivae, or all (default: all).
  --dataset DATASET      Dataset to analyze: lastfm, yelp, or all (default: all).
  --user-limit N         Experiment user limit (default: 1000).
  --item-limit N         Experiment item limit (default: 1000).
  --experiment seed_k    Seed-based experiment to analyze (required).
  -h, --help             Show this help message.
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --model)
            [[ $# -ge 2 ]] || { echo "Missing value for --model." >&2; exit 2; }
            case "$2" in
                all | neumf | multivae) model="$2" ;;
                *) echo "Unsupported model: $2" >&2; exit 2 ;;
            esac
            shift 2
            ;;
        --dataset)
            [[ $# -ge 2 ]] || { echo "Missing value for --dataset." >&2; exit 2; }
            case "$2" in
                all | lastfm | yelp) dataset="$2" ;;
                *) echo "Unsupported dataset: $2" >&2; exit 2 ;;
            esac
            shift 2
            ;;
        --user-limit | --item-limit)
            [[ $# -ge 2 ]] || { echo "Missing value for $1." >&2; exit 2; }
            [[ "$2" =~ ^[1-9][0-9]*$ ]] || { echo "$1 must be positive." >&2; exit 2; }
            if [[ "$1" == "--user-limit" ]]; then user_limit="$2"; else item_limit="$2"; fi
            shift 2
            ;;
        --experiment)
            [[ $# -ge 2 ]] || { echo "Missing value for --experiment." >&2; exit 2; }
            experiment="$2"
            shift 2
            ;;
        -h | --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            show_help >&2
            exit 2
            ;;
    esac
done

if [[ -z "$experiment" ]]; then
    echo "--experiment is required." >&2
    exit 2
fi

if [[ -n "${RECSYS_PYTHON:-}" ]]; then
    python_command=("$RECSYS_PYTHON")
elif [[ -x ".venv/bin/python" ]]; then
    python_command=(".venv/bin/python")
else
    python_command=("uv" "run" "python")
fi

exec "${python_command[@]}" -m src.scripts.evaluation.analyze_fairness \
    --model "$model" \
    --dataset "$dataset" \
    --user-limit "$user_limit" \
    --item-limit "$item_limit" \
    --experiment "$experiment"
