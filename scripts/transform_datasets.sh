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
        if (( $# > 1 )) || (( $# == 1 )) && [[ "$1" != "--use-restaurants-users-only" ]]; then
            echo "The 'all' option only accepts --use-restaurants-users-only." >&2
            exit 2
        fi
        uv run python -m src.scripts.datasets.lastfm_transform
        uv run python -m src.scripts.datasets.yelp_transform "$@"
        ;;
    *)
        echo "Usage: $0 [lastfm|yelp|all] [--use-restaurants-users-only] [transformer arguments]" >&2
        exit 2
        ;;
esac
