"""Durable, per-fold validation checkpoints for interrupted experiments."""

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

from src.utils.experiments import _atomic_json_write, _file_sha256


def recipe_hash(recipe: dict) -> str:
    encoded = json.dumps(recipe, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def temporary_checkpoint(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.stem}.",
        suffix=".tmp.pth",
    )
    os.close(descriptor)
    return Path(name)


def publish_checkpoint(temporary_path: Path, target: Path) -> str:
    """Commit model bytes before publishing their result/manifest record."""
    with temporary_path.open("rb") as checkpoint_file:
        os.fsync(checkpoint_file.fileno())
    digest = _file_sha256(temporary_path)
    os.replace(temporary_path, target)
    directory_fd = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return digest


class FoldCheckpointStore:
    def __init__(self, models_dir: Path, model: str, recipe: dict):
        self.root = models_dir / "checkpoints"
        self.model = model
        self.recipe = recipe
        self.identity = recipe_hash(recipe)
        self.recipe_path = self.root / f"{model}.recipe.json"

    def prepare(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if self.recipe_path.exists():
            try:
                with self.recipe_path.open(encoding="utf-8") as input_file:
                    saved = json.load(input_file)
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid experiment recipe: {self.recipe_path}") from error
            if saved != self.recipe:
                raise ValueError(
                    f"Inputs or methodology changed for {self.model}: {self.recipe_path}. "
                    "Remove this experiment's results manually to start a new run."
                )
            return

        if any(self.root.glob(f"fold_*/config_*/{self.model}.*")):
            raise ValueError(
                f"Missing recipe for existing {self.model} folds: {self.recipe_path}"
            )
        _atomic_json_write(self.recipe_path, self.recipe)

    def paths(self, fold: int, config_index: int) -> tuple[Path, Path]:
        directory = self.root / f"fold_{fold}" / f"config_{config_index:02d}"
        return directory / f"{self.model}.pth", directory / f"{self.model}.json"

    def load(self, fold: int, config_index: int, hyperparameters: dict, seed: int):
        checkpoint_path, result_path = self.paths(fold, config_index)
        if not result_path.is_file():
            return None
        try:
            with result_path.open(encoding="utf-8") as input_file:
                record = json.load(input_file)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(record, dict):
            return None
        if (
            record.get("recipe_sha256") != self.identity
            or record.get("hyperparameters") != hyperparameters
        ):
            return None
        if (
            record.get("model") != self.model
            or record.get("fold") != fold
            or record.get("config_index") != config_index
            or record.get("seed") != seed
        ):
            return None
        epoch = record.get("epoch")
        score = record.get("score")
        metrics = record.get("metrics")
        digest = record.get("checkpoint_sha256")
        if (
            not isinstance(epoch, int)
            or isinstance(epoch, bool)
            or epoch < 1
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not isinstance(metrics, dict)
            or not metrics
            or not all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for value in metrics.values()
            )
            or not isinstance(digest, str)
            or len(digest) != 64
            or not checkpoint_path.is_file()
        ):
            return None
        try:
            if _file_sha256(checkpoint_path) != digest:
                return None
        except OSError:
            return None
        return {
            "fold": fold,
            "epoch": epoch,
            "score": float(score),
            "metrics": metrics,
        }

    def save(
        self,
        fold: int,
        config_index: int,
        hyperparameters: dict,
        seed: int,
        result: dict,
        temporary_path: Path,
    ) -> None:
        checkpoint_path, result_path = self.paths(fold, config_index)
        digest = publish_checkpoint(temporary_path, checkpoint_path)
        _atomic_json_write(result_path, {
            "recipe_sha256": self.identity,
            "model": self.model,
            "fold": fold,
            "config_index": config_index,
            "hyperparameters": hyperparameters,
            "seed": seed,
            "epoch": result["epoch"],
            "score": result["score"],
            "metrics": result["metrics"],
            "checkpoint_sha256": digest,
        })
