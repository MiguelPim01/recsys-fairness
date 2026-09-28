"""Create and resolve immutable, versioned experiment directories."""

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from src.splitters.lastfm_cross_val import LastFMCrossValidationSplitter
from src.splitters.yelp_cross_val import YelpCrossValidationSplitter


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_PATTERN = re.compile(r"^(\d+)_exp$")
SPLITTERS = {
    "lastfm": LastFMCrossValidationSplitter,
    "yelp": YelpCrossValidationSplitter,
}


def experiment_root(user_limit: int, item_limit: int) -> Path:
    return REPOSITORY_ROOT / "results" / f"{user_limit}_{item_limit}"


def resolve_experiment(
    user_limit: int,
    item_limit: int,
    experiment: str,
) -> Path:
    if EXPERIMENT_PATTERN.fullmatch(experiment) is None:
        raise ValueError(
            "Experiment must use the NN_exp format, for example 01_exp"
        )

    path = experiment_root(user_limit, item_limit) / experiment
    if not path.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {path}")
    if not (path / "experiment.json").is_file():
        raise FileNotFoundError(f"Experiment manifest not found: {path}")
    return path


def create_experiment(user_limit: int, item_limit: int) -> Path:
    root = experiment_root(user_limit, item_limit)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".experiments.lock"

    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        indexes = [
            int(match.group(1))
            for child in root.iterdir()
            if child.is_dir()
            and (match := EXPERIMENT_PATTERN.fullmatch(child.name)) is not None
        ]
        index = max(indexes, default=0) + 1
        path = root / f"{index:02d}_exp"
        path.mkdir()
        _atomic_json_write(
            path / "experiment.json",
            {
                "experiment": path.name,
                "user_limit": user_limit,
                "item_limit": item_limit,
                "status": "created",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "datasets": {},
            },
        )

    return path


def snapshot_dataset(
    experiment_dir: Path,
    source_root: Path,
    dataset: str,
    folds: int,
    seed: int,
) -> Path:
    source_dir = source_root / dataset
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Sample directory not found: {source_dir}")

    splitter = SPLITTERS[dataset](
        dataset_dir=source_dir,
        n_splits=folds,
        seed=seed,
    )
    split_statistics = splitter.prepare()

    dataset_output_dir = experiment_dir / dataset
    snapshot_parent = dataset_output_dir / "data"
    snapshot_dir = snapshot_parent / dataset
    if snapshot_dir.exists():
        raise FileExistsError(
            f"Dataset snapshot already exists and will not be overwritten: "
            f"{snapshot_dir}"
        )

    snapshot_parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{dataset}.", suffix=".tmp", dir=snapshot_parent)
    )
    try:
        for source_path in source_dir.iterdir():
            destination = temporary_dir / source_path.name
            if source_path.is_dir():
                shutil.copytree(source_path, destination)
            else:
                shutil.copy2(source_path, destination)
        os.replace(temporary_dir, snapshot_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    manifest_path = experiment_dir / "experiment.json"
    manifest = _read_json(manifest_path)
    manifest["status"] = "prepared"
    manifest.setdefault("datasets", {})[dataset] = {
        "snapshot": str(snapshot_dir.relative_to(experiment_dir)),
        "sha256": _directory_sha256(snapshot_dir),
        "split_statistics": {
            key: value
            for key, value in split_statistics.items()
            if key != "reused"
        },
    }
    _atomic_json_write(manifest_path, manifest)
    return snapshot_dir


def set_experiment_status(experiment_dir: Path, status: str) -> None:
    manifest_path = experiment_dir / "experiment.json"
    manifest = _read_json(manifest_path)
    manifest["status"] = status
    if status == "complete":
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json_write(manifest_path, manifest)


def register_model(
    experiment_dir: Path,
    dataset: str,
    model: str,
    checkpoint_path: Path,
) -> None:
    """Register a completed checkpoint without losing parallel model updates."""
    manifest_path = experiment_dir / "experiment.json"
    lock_path = experiment_dir / ".experiment.json.lock"
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        manifest = _read_json(manifest_path)
        dataset_entry = manifest.setdefault("datasets", {}).setdefault(dataset, {})
        dataset_entry.setdefault("models", {})[model] = {
            "checkpoint": str(checkpoint_path.relative_to(experiment_dir)),
            "sha256": _file_sha256(checkpoint_path),
        }
        manifest["status"] = "trained"
        _atomic_json_write(manifest_path, manifest)


def validate_dataset_snapshot(experiment_dir: Path, dataset: str) -> Path:
    manifest = _read_json(experiment_dir / "experiment.json")
    try:
        entry = manifest["datasets"][dataset]
        snapshot_dir = experiment_dir / entry["snapshot"]
        expected_hash = entry["sha256"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"Experiment manifest has no snapshot for {dataset}"
        ) from error
    if not snapshot_dir.is_dir():
        raise FileNotFoundError(f"Dataset snapshot not found: {snapshot_dir}")
    actual_hash = _directory_sha256(snapshot_dir)
    if actual_hash != expected_hash:
        raise ValueError(
            f"Dataset snapshot checksum mismatch for {dataset}: "
            f"{actual_hash} != {expected_hash}"
        )
    return snapshot_dir


def validate_model_checkpoint(
    experiment_dir: Path,
    dataset: str,
    model: str,
) -> Path:
    manifest = _read_json(experiment_dir / "experiment.json")
    try:
        entry = manifest["datasets"][dataset]["models"][model]
        checkpoint_path = experiment_dir / entry["checkpoint"]
        expected_hash = entry["sha256"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"Experiment manifest has no {model} checkpoint for {dataset}"
        ) from error
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    actual_hash = _file_sha256(checkpoint_path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"Checkpoint checksum mismatch for {dataset}/{model}: "
            f"{actual_hash} != {expected_hash}"
        )
    return checkpoint_path


def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        candidate for candidate in path.rglob("*") if candidate.is_file()
    )
    for file_path in files:
        digest.update(str(file_path.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        with file_path.open("rb") as input_file:
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path):
    with path.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def _atomic_json_write(path: Path, value) -> None:
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)
            json.dump(value, output_file, indent=2, sort_keys=True)
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise


def _parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create")
    create.add_argument("--user-limit", type=int, required=True)
    create.add_argument("--item-limit", type=int, required=True)

    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("--user-limit", type=int, required=True)
    snapshot.add_argument("--item-limit", type=int, required=True)
    snapshot.add_argument("--experiment", required=True)
    snapshot.add_argument("--dataset", choices=("all", *SPLITTERS), default="all")
    snapshot.add_argument(
        "--source-root",
        type=Path,
        default=REPOSITORY_ROOT / "data/sample",
    )
    snapshot.add_argument("--folds", type=int, default=5)
    snapshot.add_argument("--seed", type=int, default=42)

    status = subparsers.add_parser("status")
    status.add_argument("--user-limit", type=int, required=True)
    status.add_argument("--item-limit", type=int, required=True)
    status.add_argument("--experiment", required=True)
    status.add_argument("--status", choices=("failed", "complete"), required=True)
    return parser.parse_args()


def main():
    arguments = _parse_arguments()
    if arguments.command == "create":
        path = create_experiment(arguments.user_limit, arguments.item_limit)
        print(path.name)
        return

    experiment_dir = resolve_experiment(
        arguments.user_limit,
        arguments.item_limit,
        arguments.experiment,
    )
    if arguments.command == "status":
        set_experiment_status(experiment_dir, arguments.status)
        return

    datasets = SPLITTERS if arguments.dataset == "all" else (arguments.dataset,)
    for dataset in datasets:
        path = snapshot_dataset(
            experiment_dir=experiment_dir,
            source_root=arguments.source_root,
            dataset=dataset,
            folds=arguments.folds,
            seed=arguments.seed,
        )
        print(f"{dataset}: {path}")


if __name__ == "__main__":
    main()
