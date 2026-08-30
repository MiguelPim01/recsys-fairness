#!/usr/bin/env bash

set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_root"

model="neumf"
dataset="all"
evaluation_arguments=()

show_help() {
    cat <<'EOF'
Usage: scripts/evaluate_models.sh [options]

Train and evaluate recommendation models.

Options:
  --model MODEL              Model to evaluate (default: neumf).
  --dataset DATASET          Dataset to evaluate: all, lastfm, or yelp (default: all).
  --cross-validation         Run user-stratified cross-validation.
  --hyperparameter-search    Search configurations from the model search YAML.
  --folds N                  Number of cross-validation folds (default: 5).
  -h, --help                 Show this help message.

Examples:
  scripts/evaluate_models.sh
  scripts/evaluate_models.sh --dataset yelp
  scripts/evaluate_models.sh --cross-validation
  scripts/evaluate_models.sh --hyperparameter-search
  scripts/evaluate_models.sh --cross-validation --hyperparameter-search
  scripts/evaluate_models.sh --cross-validation --folds 3
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --model)
            if (( $# < 2 )); then
                echo "Missing value for --model." >&2
                exit 2
            fi
            model="$2"
            shift 2
            ;;
        --dataset)
            if (( $# < 2 )); then
                echo "Missing value for --dataset." >&2
                exit 2
            fi
            case "$2" in
                all | lastfm | yelp)
                    dataset="$2"
                    ;;
                *)
                    echo "Unsupported dataset: $2. Available datasets: all, lastfm, yelp." >&2
                    exit 2
                    ;;
            esac
            shift 2
            ;;
        --cross-validation | --hyperparameter-search)
            evaluation_arguments+=("$1")
            shift
            ;;
        --folds)
            if (( $# < 2 )); then
                echo "Missing value for --folds." >&2
                exit 2
            fi
            if [[ ! "$2" =~ ^[0-9]+$ ]] || (( $2 < 2 )); then
                echo "--folds must be an integer greater than or equal to 2." >&2
                exit 2
            fi
            evaluation_arguments+=("--folds" "$2")
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

if [[ -x ".venv/bin/python" ]]; then
    python_command=(".venv/bin/python")
else
    python_command=("uv" "run" "python")
fi

case "$model" in
    neumf)
        exec "${python_command[@]}" -m src.scripts.evaluation.eval_neumf \
            --dataset "$dataset" \
            "${evaluation_arguments[@]}"
        ;;
    *)
        echo "Unsupported model: $model. Available models: neumf." >&2
        exit 2
        ;;
esac
