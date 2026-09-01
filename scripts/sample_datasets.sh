#!/usr/bin/env bash

set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_root"

dataset="${1:-all}"
if (( $# > 0 )); then
    shift
fi

if [[ -x ".venv/bin/python" ]]; then
    python_command=(".venv/bin/python")
else
    python_command=("uv" "run" "python")
fi

case "$dataset" in
    lastfm)
        exec "${python_command[@]}" -m src.scripts.datasets.lastfm_sample "$@"
        ;;
    yelp)
        exec "${python_command[@]}" -m src.scripts.datasets.yelp_sample "$@"
        ;;
    all)
        "${python_command[@]}" -m src.scripts.datasets.lastfm_sample "$@"
        "${python_command[@]}" -m src.scripts.datasets.yelp_sample "$@"
        ;;
    *)
        echo "Usage: $0 [lastfm|yelp|all] [sampler arguments]" >&2
        exit 2
        ;;
esac
