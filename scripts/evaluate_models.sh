#!/usr/bin/env bash

set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_root"

model="neumf"
dataset="all"
user_limit=1000
item_limit=1000
evaluation_arguments=()

show_help() {
    cat <<'EOF'
Usage: scripts/evaluate_models.sh [options]

Train and evaluate recommendation models.

Options:
  --model MODEL              Model to evaluate: neumf, multivae, or all (default: neumf).
  --dataset DATASET          Dataset to evaluate: all, lastfm, or yelp (default: all).
  --user-limit N             Number of sampled users (default: 1000).
  --item-limit N             Number of sampled items (default: 1000).
  --cross-validation         Run user-stratified cross-validation.
  --hyperparameter-search    Search configurations from the model search YAML.
  --folds N                  Number of cross-validation folds (default: 5).
  -h, --help                 Show this help message.

Examples:
  scripts/evaluate_models.sh
  scripts/evaluate_models.sh --model multivae
  scripts/evaluate_models.sh --model all --dataset yelp
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
        --user-limit | --item-limit)
            if (( $# < 2 )); then
                echo "Missing value for $1." >&2
                exit 2
            fi

            if [[ ! "$2" =~ ^[1-9][0-9]*$ ]]; then
                echo "$1 must be a positive integer." >&2
                exit 2
            fi

            if [[ "$1" == "--user-limit" ]]; then
                user_limit="$2"
            else
                item_limit="$2"
            fi

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

evaluation_arguments+=(
    --user-limit "$user_limit"
    --item-limit "$item_limit"
)

case "$model" in
    neumf)
        exec "${python_command[@]}" -m src.scripts.evaluation.eval_neumf \
            --dataset "$dataset" \
            "${evaluation_arguments[@]}"
        ;;
    multivae)
        exec "${python_command[@]}" -m src.scripts.evaluation.eval_multivae \
            --dataset "$dataset" \
            "${evaluation_arguments[@]}"
        ;;
    all)
        "${python_command[@]}" -m src.scripts.evaluation.eval_neumf \
            --dataset "$dataset" \
            "${evaluation_arguments[@]}"
        "${python_command[@]}" -m src.scripts.evaluation.eval_multivae \
            --dataset "$dataset" \
            "${evaluation_arguments[@]}"
        ;;
    *)
        echo "Unsupported model: $model. Available models: neumf, multivae, all." >&2
        exit 2
        ;;
esac
