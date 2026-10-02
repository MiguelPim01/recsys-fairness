"""Create and resolve immutable experiments identified by their random seed."""

import argparse
import fcntl
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from src.splitters.lastfm_cross_val import LastFMCrossValidationSplitter
from src.splitters.yelp_cross_val import YelpCrossValidationSplitter

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_PATTERN = re.compile(r"^seed_(\d+)$")
LEGACY_EXPERIMENT_PATTERN = re.compile(r"^\d+_exp$")
SPLITTERS = {
    "lastfm": LastFMCrossValidationSplitter,
    "yelp": YelpCrossValidationSplitter,
}


def experiment_root(user_limit: int, item_limit: int) -> Path:
    return REPOSITORY_ROOT / "results" / f"{user_limit}_{item_limit}"


def sample_root(user_limit: int, item_limit: int, experiment: str) -> Path:
    return REPOSITORY_ROOT / "data" / "sample" / f"{user_limit}_{item_limit}" / experiment


def resolve_experiment(user_limit: int, item_limit: int, experiment: str) -> Path:
    if (
        EXPERIMENT_PATTERN.fullmatch(experiment) is None
        and LEGACY_EXPERIMENT_PATTERN.fullmatch(experiment) is None
    ):
        raise ValueError(
            "Experiment must use the seed_k format, for example seed_42"
        )

    path = experiment_root(user_limit, item_limit) / experiment
    if not path.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {path}")
    if not (path / "experiment.json").is_file():
        raise FileNotFoundError(f"Experiment manifest not found: {path}")
    
    return path


def create_experiment(user_limit: int, item_limit: int, seed: int) -> Path:
    if seed < 0:
        raise ValueError("Seed must be a non-negative integer")

    root = experiment_root(user_limit, item_limit)
    root.mkdir(parents=True, exist_ok=True)

    path = root / f"seed_{seed}"
    try:
        path.mkdir()
    except FileExistsError:
        raise FileExistsError(
            f"Experiment already exists and will not be overwritten: {path}"
        ) from None
    _atomic_json_write(
        path / "experiment.json",
        {
            "experiment": path.name,
            "seed": seed,
            "user_limit": user_limit,
            "item_limit": item_limit,
            "status": "created",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "datasets": {},
        },
    )

    return path


def create_or_resume_experiment(user_limit: int, item_limit: int, seed: int) -> Path:
    if seed < 0:
        raise ValueError("Seed must be a non-negative integer")

    path = experiment_root(user_limit, item_limit) / f"seed_{seed}"
    if path.is_dir() and (path / "experiment.json").is_file():
        return path
    return create_experiment(user_limit, item_limit, seed)


def experiment_status(experiment_dir: Path) -> str:
    """Return the recorded status string of an experiment."""
    manifest = _read_json(experiment_dir / "experiment.json")
    return str(manifest.get("status", "unknown"))


def is_experiment_complete(user_limit: int, item_limit: int, seed: int) -> bool:
    """Return True when the seed exists and is marked ``complete``."""
    path = experiment_root(user_limit, item_limit) / f"seed_{seed}"
    manifest_path = path / "experiment.json"
    if not manifest_path.is_file():
        return False
    try:
        return _read_json(manifest_path).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def is_model_trained(experiment_dir: Path, dataset: str, model: str) -> bool:
    """Return True when a model checkpoint is registered AND verifies on disk."""
    try:
        validate_model_checkpoint(experiment_dir, dataset, model)
        return True
    except (ValueError, FileNotFoundError):
        return False


def discard_orphan_checkpoint(checkpoint_path: Path) -> None:
    """Remove a checkpoint file left behind by an interrupted training run."""
    try:
        checkpoint_path.unlink()
    except FileNotFoundError:
        pass


def experiment_seed(experiment_dir: Path) -> int:
    manifest = _read_json(experiment_dir / "experiment.json")
    seed = manifest.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        if LEGACY_EXPERIMENT_PATTERN.fullmatch(experiment_dir.name):
            return 42
        raise ValueError(f"Experiment manifest has an invalid seed: {experiment_dir}")
    return seed


def prepare_dataset(experiment_dir: Path, dataset: str, folds: int) -> Path:
    manifest_path = experiment_dir / "experiment.json"
    manifest = _read_json(manifest_path)
    source_dir = sample_root(
        manifest["user_limit"],
        manifest["item_limit"],
        experiment_dir.name,
    ) / dataset
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Sample directory not found: {source_dir}")

    splitter = SPLITTERS[dataset](
        dataset_dir=source_dir,
        n_splits=folds,
        seed=experiment_seed(experiment_dir),
    )
    split_statistics = splitter.prepare()

    manifest["status"] = "prepared"
    manifest.setdefault("datasets", {})[dataset] = {
        "sample": str(source_dir.relative_to(REPOSITORY_ROOT)),
        "sha256": _directory_sha256(source_dir),
        "split_statistics": {
            key: value
            for key, value in split_statistics.items()
            if key != "reused"
        },
    }
    _atomic_json_write(manifest_path, manifest)
    return source_dir


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


def validate_experiment_dataset(experiment_dir: Path, dataset: str) -> Path:
    manifest = _read_json(experiment_dir / "experiment.json")
    try:
        entry = manifest["datasets"][dataset]
        expected_hash = entry["sha256"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"Experiment manifest has no dataset for {dataset}"
        ) from error

    if "sample" in entry:
        dataset_dir = REPOSITORY_ROOT / entry["sample"]
    elif "snapshot" in entry:
        dataset_dir = experiment_dir / entry["snapshot"]
    else:
        raise ValueError(f"Experiment manifest has no dataset path for {dataset}")

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Experiment dataset not found: {dataset_dir}")
    actual_hash = _directory_sha256(dataset_dir)
    if actual_hash != expected_hash:
        raise ValueError(
            f"Dataset checksum mismatch for {dataset}: "
            f"{actual_hash} != {expected_hash}"
        )
    return dataset_dir


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
    create.add_argument("--seed", type=int, default=42)

    create_or_resume = subparsers.add_parser("create-or-resume")
    create_or_resume.add_argument("--user-limit", type=int, required=True)
    create_or_resume.add_argument("--item-limit", type=int, required=True)
    create_or_resume.add_argument("--seed", type=int, default=42)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--user-limit", type=int, required=True)
    prepare.add_argument("--item-limit", type=int, required=True)
    prepare.add_argument("--experiment", required=True)
    prepare.add_argument("--dataset", choices=("all", *SPLITTERS), default="all")
    prepare.add_argument("--folds", type=int, default=5)

    status = subparsers.add_parser("status")
    status.add_argument("--user-limit", type=int, required=True)
    status.add_argument("--item-limit", type=int, required=True)
    status.add_argument("--experiment", required=True)
    status.add_argument("--status", choices=("failed", "complete"), required=True)
    return parser.parse_args()


def main():
    arguments = _parse_arguments()
    if arguments.command == "create":
        path = create_experiment(
            arguments.user_limit,
            arguments.item_limit,
            arguments.seed,
        )
        print(path.name)
        return

    if arguments.command == "create-or-resume":
        path = create_or_resume_experiment(
            arguments.user_limit,
            arguments.item_limit,
            arguments.seed,
        )
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
        path = prepare_dataset(
            experiment_dir=experiment_dir,
            dataset=dataset,
            folds=arguments.folds,
        )
        print(f"{dataset}: {path}")


if __name__ == "__main__":
    main()
