import argparse
import os
import subprocess
import sys
from pathlib import Path

from src.utils.experiments import (
    create_or_resume_experiment,
    experiment_status,
    is_experiment_complete,
    resolve_experiment,
    set_experiment_status,
)
from src.utils.transforms import can_reuse_transformation

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPOSITORY_ROOT / "scripts"


def parse_seed_list(raw: str) -> list[int]:
    """Parse a seed specification into a sorted, de-duplicated list.

    Accepts comma/space separated integers and inclusive ranges written as
    ``a-b``. Examples: ``"42"``, ``"42,43,44"``, ``"42-46"``, ``"42-44,50"``.
    """
    seeds: set[int] = set()
    for token in raw.replace(",", " ").split():
        if "-" in token and not token.startswith("-"):
            start_text, _, end_text = token.partition("-")
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid seed range (end < start): {token}")
            seeds.update(range(start, end + 1))
        else:
            seeds.add(int(token))
    if not seeds:
        raise ValueError("Seed list is empty")
    if any(seed < 0 for seed in seeds):
        raise ValueError("Seeds must be non-negative integers")
    return sorted(seeds)


def seeds_for_worker(
    seeds: list[int], worker_index: int, worker_count: int
) -> list[int]:
    """Return the slice of seeds assigned to one worker (index % count)."""
    if worker_count < 1:
        raise ValueError("worker_count must be >= 1")
    if not 0 <= worker_index < worker_count:
        raise ValueError("worker_index must be in [0, worker_count)")
    return [
        seed
        for position, seed in enumerate(seeds)
        if position % worker_count == worker_index
    ]


def _python_command() -> list[str]:
    """Select the interpreter, matching the scripts' own resolution order."""
    override = os.environ.get("RECSYS_PYTHON")
    if override:
        return [override]
    venv_python = REPOSITORY_ROOT / ".venv" / "bin" / "python"
    if venv_python.is_file() and os.access(venv_python, os.X_OK):
        return [str(venv_python)]
    return ["uv", "run", "python"]


def _run(command: list[str]) -> None:
    """Run a pipeline step, streaming output, aborting on failure."""
    print(f"\n$ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=str(REPOSITORY_ROOT), check=True)


def _require_transformed_datasets(use_restaurants_users_only: bool) -> None:
    """Fail before sampling if the manually prepared datasets are unavailable."""
    variants = {
        "lastfm": {"format": "default"},
        "yelp": {"use_restaurants_users_only": use_restaurants_users_only},
    }
    for dataset, variant in variants.items():
        output_dir = REPOSITORY_ROOT / "data" / "processed" / dataset
        if not can_reuse_transformation(output_dir, dataset, variant):
            option = (
                " --use-restaurants-users-only"
                if use_restaurants_users_only else ""
            )
            raise RuntimeError(
                f"Processed {dataset} data is missing, incomplete, or uses a "
                f"different variant: {output_dir}. Run "
                f"'./scripts/transform_datasets.sh all{option}' locally first; "
                "for Slurm, copy data/processed/ to the project on the NFS."
            )


def run_seed(
    seed: int,
    user_limit: int,
    item_limit: int,
    folds: int,
    fold_workers: int,
    use_restaurants_users_only: bool,
) -> None:
    """Run (or resume) the full pipeline for a single seed."""
    if is_experiment_complete(user_limit, item_limit, seed):
        print(f"[seed {seed}] already complete; skipping.", flush=True)
        return

    _require_transformed_datasets(use_restaurants_users_only)

    experiment_dir = create_or_resume_experiment(user_limit, item_limit, seed)
    experiment = experiment_dir.name
    print(
        f"[seed {seed}] starting (experiment={experiment}, "
        f"status={experiment_status(experiment_dir)})",
        flush=True,
    )

    python = _python_command()
    sample_sh = str(SCRIPTS_DIR / "sample_datasets.sh")
    evaluate_sh = str(SCRIPTS_DIR / "evaluate_models.sh")
    fairness_sh = str(SCRIPTS_DIR / "analyze_fairness.sh")

    try:
        # 1. Sample the previously transformed datasets (reused when valid).
        _run([
            sample_sh, "all",
            "--user-limit", str(user_limit),
            "--item-limit", str(item_limit),
            "--experiment", experiment,
        ])

        # 2. Prepare k-fold splits (reused when the split manifest matches).
        _run([
            *python, "-m", "src.utils.experiments", "prepare",
            "--user-limit", str(user_limit),
            "--item-limit", str(item_limit),
            "--experiment", experiment,
            "--dataset", "all",
            "--folds", str(folds),
        ])

        # 3. Train + evaluate every model (already-trained models are skipped).
        _run([
            evaluate_sh,
            "--model", "all",
            "--dataset", "all",
            "--experiment", experiment,
            "--user-limit", str(user_limit),
            "--item-limit", str(item_limit),
            "--cross-validation",
            "--hyperparameter-search",
            "--folds", str(folds),
            "--fold-workers", str(fold_workers),
        ])

        # 4. Fairness analysis from the saved checkpoints (safe to rerun).
        _run([
            fairness_sh,
            "--model", "all",
            "--dataset", "all",
            "--experiment", experiment,
            "--user-limit", str(user_limit),
            "--item-limit", str(item_limit),
        ])
    except BaseException:
        # Record the interruption but keep every artifact so the next run
        # resumes from here instead of restarting.
        set_experiment_status(experiment_dir, "failed")
        raise

    set_experiment_status(experiment_dir, "complete")
    print(f"[seed {seed}] complete.", flush=True)


def run_campaign(
    seeds: list[int],
    worker_index: int,
    worker_count: int,
    user_limit: int,
    item_limit: int,
    folds: int,
    fold_workers: int,
    use_restaurants_users_only: bool,
) -> None:
    assigned = seeds_for_worker(seeds, worker_index, worker_count)
    print(
        f"Worker {worker_index}/{worker_count} owns seeds: "
        f"{assigned or '(none)'}",
        flush=True,
    )
    for seed in assigned:
        run_seed(
            seed=seed,
            user_limit=user_limit,
            item_limit=item_limit,
            folds=folds,
            fold_workers=fold_workers,
            use_restaurants_users_only=use_restaurants_users_only,
        )
    print(
        f"Worker {worker_index}/{worker_count} finished its assigned seeds.",
        flush=True,
    )


def count_pending(
    seeds: list[int],
    worker_index: int,
    worker_count: int,
    user_limit: int,
    item_limit: int,
) -> int:
    assigned = seeds_for_worker(seeds, worker_index, worker_count)
    return sum(
        1
        for seed in assigned
        if not is_experiment_complete(user_limit, item_limit, seed)
    )


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--seeds",
        required=True,
        help="Seeds to run: integers and ranges, e.g. '42-46' or '42,43,50'.",
    )
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--user-limit", type=int, default=1000)
    parser.add_argument("--item-limit", type=int, default=1000)


def _parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run this worker's assigned seeds.")
    _add_common_arguments(run)
    run.add_argument("--folds", type=int, default=5)
    run.add_argument("--fold-workers", type=int, default=1)
    run.add_argument(
        "--use-restaurants-users-only",
        action="store_true",
        help="Keep only Yelp users whose predominant preference is restaurants.",
    )

    pending = subparsers.add_parser(
        "pending",
        help="Print the number of this worker's seeds that are not complete.",
    )
    _add_common_arguments(pending)

    status = subparsers.add_parser(
        "status",
        help="Print a per-seed status table for the whole campaign.",
    )
    _add_common_arguments(status)

    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    seeds = parse_seed_list(arguments.seeds)

    if arguments.command == "run":
        run_campaign(
            seeds=seeds,
            worker_index=arguments.worker_index,
            worker_count=arguments.worker_count,
            user_limit=arguments.user_limit,
            item_limit=arguments.item_limit,
            folds=arguments.folds,
            fold_workers=arguments.fold_workers,
            use_restaurants_users_only=arguments.use_restaurants_users_only,
        )
        return

    if arguments.command == "pending":
        print(
            count_pending(
                seeds=seeds,
                worker_index=arguments.worker_index,
                worker_count=arguments.worker_count,
                user_limit=arguments.user_limit,
                item_limit=arguments.item_limit,
            )
        )
        return

    if arguments.command == "status":
        assigned = seeds_for_worker(
            seeds, arguments.worker_index, arguments.worker_count
        )
        for seed in assigned:
            if is_experiment_complete(
                arguments.user_limit, arguments.item_limit, seed
            ):
                state = "complete"
            else:
                try:
                    experiment_dir = resolve_experiment(
                        arguments.user_limit,
                        arguments.item_limit,
                        f"seed_{seed}",
                    )
                    state = experiment_status(experiment_dir)
                except (FileNotFoundError, ValueError):
                    state = "not-started"
            print(f"seed_{seed}\t{state}")
        return


if __name__ == "__main__":
    sys.exit(main())
