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
    _atomic_json_write(path / "experiment.json", _new_manifest(path, user_limit, item_limit, seed))

    return path


def create_or_resume_experiment(user_limit: int, item_limit: int, seed: int) -> Path:
    if seed < 0:
        raise ValueError("Seed must be a non-negative integer")

    path = experiment_root(user_limit, item_limit) / f"seed_{seed}"
    if path.is_dir():
        manifest_path = path / "experiment.json"
        if manifest_path.is_file():
            manifest = _read_json(manifest_path)
            if (
                not isinstance(manifest, dict)
                or manifest.get("seed") != seed
                or manifest.get("user_limit") != user_limit
                or manifest.get("item_limit") != item_limit
            ):
                raise ValueError(f"Experiment identity mismatch: {manifest_path}")
            return path
        lock_path = path / ".experiment.json.lock"
        with lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            if manifest_path.is_file():
                return path
            if any(
                entry.name != lock_path.name and not (
                    entry.name.startswith(".experiment.json.")
                    and entry.name.endswith(".tmp")
                )
                for entry in path.iterdir()
            ):
                raise ValueError(
                    f"Experiment has files but no manifest: {path}. "
                    "Inspect it before resuming."
                )
            _atomic_json_write(
                manifest_path, _new_manifest(path, user_limit, item_limit, seed)
            )
        return path
    return create_experiment(user_limit, item_limit, seed)


def _new_manifest(path: Path, user_limit: int, item_limit: int, seed: int) -> dict:
    return {
        "experiment": path.name,
        "seed": seed,
        "user_limit": user_limit,
        "item_limit": item_limit,
        "status": "created",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "datasets": {},
    }


def experiment_status(experiment_dir: Path) -> str:
    """Return the recorded status string of an experiment."""
    manifest = _read_json(experiment_dir / "experiment.json")
    return str(manifest.get("status", "unknown"))


def is_experiment_complete(user_limit: int, item_limit: int, seed: int) -> bool:
    """Return True only when the completed campaign's artifacts still verify."""
    path = experiment_root(user_limit, item_limit) / f"seed_{seed}"
    manifest_path = path / "experiment.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = _read_json(manifest_path)
        if not isinstance(manifest, dict) or manifest.get("status") != "complete":
            return False
        for dataset in SPLITTERS:
            for model in ("neumf", "multivae"):
                validate_model_checkpoint(path, dataset, model)
            dataset_dir = path / dataset
            fairness = _read_json(dataset_dir / "results.json")
            if (
                not isinstance(fairness, dict)
                or not isinstance(fairness.get("results"), dict)
                or not {"NeuMF", "MultiVAE"}.issubset(fairness["results"])
            ):
                return False
            _read_json(dataset_dir / "k_clusters_fairness.json")
            for artifact in (
                "boxplot.pdf",
                "grp_unfairness_by_model_and_groups.pdf",
                "grp_loss_by_model_and_groups.pdf",
                "grp_unfairness_and_error_table.json",
            ):
                output_path = dataset_dir / artifact
                if not output_path.is_file() or output_path.stat().st_size == 0:
                    return False
        return True
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
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


def ensure_recorded_split_reusable(experiment_dir: Path, dataset: str, splitter) -> None:
    """Protect an experiment's recorded split files from regeneration."""
    manifest = _read_json(experiment_dir / "experiment.json")
    dataset_entry = manifest.get("datasets", {}).get(dataset)
    if not isinstance(dataset_entry, dict) or not dataset_entry.get("sha256"):
        return

    split_manifest = splitter._read_manifest()
    source_hash = splitter._sha256(splitter.interaction_path)
    if not splitter._can_reuse(split_manifest, source_hash, splitter._expected_files()):
        raise ValueError(
            f"Split configuration or files changed for {dataset} in {experiment_dir}. "
            "Start a new experiment with a different seed or limits to use "
            "the current split method."
        )


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
    dataset_entry = manifest.get("datasets", {}).get(dataset)
    ensure_recorded_split_reusable(experiment_dir, dataset, splitter)
    split_statistics = splitter.prepare()

    sample_path = str(source_dir.relative_to(REPOSITORY_ROOT))
    sample_hash = _directory_sha256(source_dir)
    if (
        isinstance(dataset_entry, dict)
        and dataset_entry.get("sample") == sample_path
        and dataset_entry.get("sha256") == sample_hash
    ):
        return source_dir

    if isinstance(dataset_entry, dict) and dataset_entry.get("sha256"):
        raise ValueError(
            f"Sample or splits changed for {dataset} in {experiment_dir}. "
            "Remove the experiment results manually before starting a new run."
        )

    lock_path = experiment_dir / ".experiment.json.lock"
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        manifest = _read_json(manifest_path)
        dataset_entry = manifest.get("datasets", {}).get(dataset)
        if isinstance(dataset_entry, dict) and dataset_entry.get("sha256"):
            if (
                dataset_entry.get("sample") == sample_path
                and dataset_entry.get("sha256") == sample_hash
            ):
                return source_dir
            raise ValueError(f"Sample or splits changed for {dataset} in {experiment_dir}")
        manifest["status"] = "prepared"
        manifest.setdefault("datasets", {})[dataset] = {
            "sample": sample_path,
            "sha256": sample_hash,
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
    lock_path = experiment_dir / ".experiment.json.lock"
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
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
        candidate for candidate in path.rglob("*")
        if candidate.is_file()
        and not (candidate.name.startswith(".") and candidate.name.endswith(".tmp"))
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
            json.dump(value, output_file, indent=2, sort_keys=True, allow_nan=False)
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
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
