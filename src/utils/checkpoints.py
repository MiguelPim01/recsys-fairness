"""Durable fold results and final-model checkpoints for interrupted experiments."""

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

from src.utils.experiments import _atomic_json_write, _file_sha256

LEGACY_EVALUATOR_SHA256 = "4555964934ce75ca3fd410a6e2a9b5bcc3720ac5913bec2ce534f03472e040bd"
TRACKING_EVALUATOR_SHA256 = "f4748f8f6e930a1ee61e9bbfaaaa63e25fbbb7f4d550399a210263722abedf3e"
JSON_ONLY_EVALUATOR_SHA256 = "4b2e26be3c363ee846901e4c82a0719c2dfde185f5a3ffb2cd199b3ea800a19f"


def _compatible_recipe_change(saved: dict, current: dict) -> bool:
    old_sources = saved.get("source_sha256")
    new_sources = current.get("source_sha256")
    if not isinstance(old_sources, dict) or not isinstance(new_sources, dict):
        return False
    # Only audited changes to tracking and fold persistence may reuse results.
    compatible_sources = (
        (LEGACY_EVALUATOR_SHA256, TRACKING_EVALUATOR_SHA256),
        (LEGACY_EVALUATOR_SHA256, JSON_ONLY_EVALUATOR_SHA256),
        (TRACKING_EVALUATOR_SHA256, JSON_ONLY_EVALUATOR_SHA256),
    )
    source_change = (
        old_sources.get("IModelEvaluator"), new_sources.get("IModelEvaluator")
    )
    if source_change not in compatible_sources:
        return False
    return (
        {**saved, "source_sha256": {**old_sources, "IModelEvaluator": None}}
        == {**current, "source_sha256": {**new_sources, "IModelEvaluator": None}}
    )


def _valid_fold_result(result: dict) -> bool:
    epoch = result.get("epoch")
    score = result.get("score")
    metrics = result.get("metrics")
    return (
        type(epoch) is int
        and epoch >= 1
        and type(score) in (int, float)
        and math.isfinite(score)
        and isinstance(metrics, dict)
        and bool(metrics)
        and all(
            type(value) in (int, float) and math.isfinite(value)
            for value in metrics.values()
        )
    )


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
            if not isinstance(saved, dict):
                raise ValueError(f"Invalid experiment recipe: {self.recipe_path}")
            if saved != self.recipe:
                if _compatible_recipe_change(saved, self.recipe):
                    # Fold JSONs retain the original recipe hash; keep using it.
                    self.recipe = saved
                    self.identity = recipe_hash(saved)
                else:
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
        _, result_path = self.paths(fold, config_index)
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
            or any(
                type(record.get(key)) is not int
                for key in ("fold", "config_index", "seed")
            )
        ):
            return None
        if not _valid_fold_result(record):
            return None
        return {
            "fold": fold,
            "epoch": record["epoch"],
            "score": float(record["score"]),
            "metrics": record["metrics"],
        }

    def save(
        self,
        fold: int,
        config_index: int,
        hyperparameters: dict,
        seed: int,
        result: dict,
    ) -> None:
        if not _valid_fold_result(result):
            raise ValueError(f"Invalid validation result for {self.model} fold {fold}")
        _, result_path = self.paths(fold, config_index)
        result_path.parent.mkdir(parents=True, exist_ok=True)
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
        })
