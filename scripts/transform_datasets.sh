#!/usr/bin/env bash

set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_root"

dataset="${1:-all}"
if (( $# > 0 )); then
    shift
fi

if [[ -n "${RECSYS_PYTHON:-}" ]]; then
    python_command=("$RECSYS_PYTHON")
elif [[ -x ".venv/bin/python" ]]; then
    python_command=(".venv/bin/python")
else
    python_command=("uv" "run" "python")
fi

case "$dataset" in
    lastfm)
        exec "${python_command[@]}" -m src.scripts.datasets.lastfm_transform "$@"
        ;;
    yelp)
        exec "${python_command[@]}" -m src.scripts.datasets.yelp_transform "$@"
        ;;
    all)
        if (( $# > 1 )) || (( $# == 1 )) && [[ "$1" != "--use-restaurants-users-only" ]]; then
            echo "The 'all' option only accepts --use-restaurants-users-only." >&2
            exit 2
        fi
        "${python_command[@]}" -m src.scripts.datasets.lastfm_transform
        "${python_command[@]}" -m src.scripts.datasets.yelp_transform "$@"
        ;;
    *)
        echo "Usage: $0 [lastfm|yelp|all] [--use-restaurants-users-only] [transformer arguments]" >&2
        exit 2
        ;;
esac
