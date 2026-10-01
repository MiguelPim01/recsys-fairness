"""Validate reusable transformed datasets and record transformation variants."""

import json
import os
import tempfile
from pathlib import Path

MANIFEST_VERSION = 1


def required_atomic_files(output_dir: Path, dataset: str) -> tuple[Path, ...]:
    return tuple(output_dir / f"{dataset}.{suffix}" for suffix in ("inter", "item", "user"))


def transformed_files_exist(output_dir: Path, dataset: str) -> bool:
    return all(path.is_file() for path in required_atomic_files(output_dir, dataset))


def transform_manifest_path(output_dir: Path, dataset: str) -> Path:
    return output_dir / f"{dataset}.transform_manifest.json"


def read_transform_variant(output_dir: Path, dataset: str):
    path = transform_manifest_path(output_dir, dataset)
    try:
        with path.open(encoding="utf-8") as input_file:
            manifest = json.load(input_file)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Transformation manifest not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid transformation manifest: {path}") from error

    if manifest.get("version") != MANIFEST_VERSION or manifest.get("dataset") != dataset:
        raise ValueError(f"Incompatible transformation manifest: {path}")
    return manifest.get("variant")


def can_reuse_transformation(output_dir: Path, dataset: str, variant) -> bool:
    if not transformed_files_exist(output_dir, dataset):
        return False
    try:
        return read_transform_variant(output_dir, dataset) == variant
    except (FileNotFoundError, ValueError):
        return False


def invalidate_transform_manifest(output_dir: Path, dataset: str) -> None:
    transform_manifest_path(output_dir, dataset).unlink(missing_ok=True)


def write_transform_manifest(output_dir: Path, dataset: str, variant) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = transform_manifest_path(output_dir, dataset)
    document = {
        "version": MANIFEST_VERSION,
        "dataset": dataset,
        "variant": variant,
    }

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_dir,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)
            json.dump(document, output_file, indent=2, sort_keys=True)
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise
    return path
