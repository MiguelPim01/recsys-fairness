#!/usr/bin/env bash

set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_root"

dataset="${1:-all}"
if (( $# > 0 )); then
    shift
fi

case "$dataset" in
    lastfm)
        exec uv run python -m src.scripts.datasets.lastfm_transform "$@"
        ;;
    yelp)
        exec uv run python -m src.scripts.datasets.yelp_transform "$@"
        ;;
    all)
        if (( $# > 0 )); then
            echo "The 'all' option does not accept transformer arguments." >&2
            exit 2
        fi
        uv run python -m src.scripts.lastfm_transform
        uv run python -m src.scripts.yelp_transform
        ;;
    *)
        echo "Usage: $0 [lastfm|yelp|all] [transformer arguments]" >&2
        exit 2
        ;;
esac
