import fcntl
import json
import math
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class ResultsStore:
    """Atomically merge one algorithm result into a dataset result file."""

    def __init__(self, output_dir: str | Path):
        output_dir = Path(output_dir)
        self.output_dir = output_dir if output_dir.is_absolute() else REPOSITORY_ROOT / output_dir

    def update(
        self,
        dataset: str,
        algorithm: str,
        analysis: dict[str, Any],
        after_update: Callable[[Path], Any] | None = None,
    ):
        """
        Update the results file for a given dataset with the analysis results of a specific algorithm.

        Args:
            dataset: Dataset name.
            algorithm: Algorithm name.
            analysis: Analysis results to be added to the results file.
            after_update: Function executed while the dataset results are locked.

        Returns:
            output_path: Path to the updated results file.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = self.output_dir / "results.json"
        lock_path = self.output_dir / f".{output_path.name}.lock"

        with lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

            document = self._read(output_path)
            document["results"][algorithm] = analysis

            temporary_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=self.output_dir,
                    prefix=f".{output_path.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as output_file:
                    temporary_path = Path(output_file.name)

                    json.dump(
                        document,
                        output_file,
                        indent=2,
                        sort_keys=True,
                        allow_nan=False,
                    )

                    output_file.write("\n")
                    output_file.flush()

                    os.fsync(output_file.fileno())

                os.replace(temporary_path, output_path)

                if after_update is not None:
                    after_update(output_path)
            except Exception:
                if temporary_path is not None and temporary_path.exists():
                    temporary_path.unlink()
                raise

        return output_path

    @staticmethod
    def _read(path: Path):
        if not path.exists():
            return {"results": {}}

        with path.open(encoding="utf-8") as input_file:
            document = json.load(input_file)

        return document


class KClustersFairnessStore:
    """Atomically merge one recommender into the cluster sensitivity tables."""

    MODEL_PRIORITY = {"NeuMF": 0, "MultiVAE": 1}

    def __init__(self, output_dir: str | Path):
        output_dir = Path(output_dir)
        self.output_dir = (
            output_dir if output_dir.is_absolute() else REPOSITORY_ROOT / output_dir
        )

    def update(self, dataset: str, model: str, values: dict[str, dict[int, float]]):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "k_clusters_fairness.json"
        lock_path = self.output_dir / ".k_clusters_fairness.json.lock"

        with lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            document = self._read(output_path, dataset)

            for clustering_name in ("kmeans", "agglomerative"):
                try:
                    clustering_values = values[clustering_name]
                except KeyError as error:
                    raise ValueError(
                        f"Missing cluster sensitivity values for {clustering_name}"
                    ) from error

                table = document["tables"].setdefault(
                    clustering_name,
                    {"columns": ["clusters"], "rows": []},
                )
                rows_by_k = {row["clusters"]: row for row in table["rows"]}
                for raw_k, raw_value in clustering_values.items():
                    k = int(raw_k)
                    value = float(raw_value)
                    if k < 2 or not math.isfinite(value) or value < 0.0:
                        raise ValueError(
                            f"Invalid {clustering_name} sensitivity for k={raw_k}"
                        )
                    rows_by_k.setdefault(k, {"clusters": k})[model] = value

                models = {
                    key
                    for row in rows_by_k.values()
                    for key in row
                    if key != "clusters"
                }
                table["columns"] = ["clusters", *sorted(models, key=self._model_key)]
                table["rows"] = [rows_by_k[k] for k in sorted(rows_by_k)]

            self._atomic_write(output_path, document)

        return output_path

    @staticmethod
    def _read(path: Path, dataset: str):
        if not path.exists():
            return {
                "dataset": dataset,
                "metric": "rgrp",
                "tables": {},
            }
        with path.open(encoding="utf-8") as input_file:
            document = json.load(input_file)
        if document.get("dataset") != dataset:
            raise ValueError(
                f"Cluster fairness dataset mismatch: "
                f"{document.get('dataset')} != {dataset}"
            )
        if document.get("metric") != "rgrp" or not isinstance(
            document.get("tables"), dict
        ):
            raise ValueError(f"Invalid cluster fairness JSON: {path}")
        return document

    @classmethod
    def _model_key(cls, model: str):
        return (
            cls.MODEL_PRIORITY.get(model, len(cls.MODEL_PRIORITY)),
            model.casefold(),
        )

    @staticmethod
    def _atomic_write(path: Path, document):
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
                json.dump(
                    document,
                    output_file,
                    indent=2,
                    sort_keys=False,
                    allow_nan=False,
                )
                output_file.write("\n")
                output_file.flush()
                os.fsync(output_file.fileno())
            os.replace(temporary_path, path)
        except Exception:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
            raise
