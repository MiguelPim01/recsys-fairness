"""Generate publication artifacts exclusively from an evaluation results JSON."""

import argparse
import json
import math
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "recsys-fairness-matplotlib")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from src.utils.console import ConsoleColor, styled_print

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
GROUPING_ORDER_BY_DATASET = {
    "lastfm": ("activity", "age", "gender", "kmeans", "agglomerative"),
    "yelp": (
        "activity",
        "friend_count",
        "fans",
        "tenure",
        "kmeans",
        "agglomerative",
    ),
}
GROUPING_LABELS = {
    "activity": "Activity",
    "age": "Age",
    "gender": "Gender",
    "friend_count": "Friends",
    "fans": "Fans",
    "tenure": "Tenure",
    "kmeans": "K-Means",
    "agglomerative": "Agglomerative",
}
GROUPING_COLORS = {
    "activity": "#ff7f0e",
    "age": "#9467bd",
    "gender": "#17becf",
    "friend_count": "#8c564b",
    "fans": "#bcbd22",
    "tenure": "#e377c2",
    "kmeans": "#1f77b4",
    "agglomerative": "#2ca02c",
}
MODEL_PRIORITY = {"NeuMF": 0, "MultiVAE": 1}
PDF_METADATA = {
    "Creator": "recsys_fairness",
    "Producer": "Matplotlib",
    "CreationDate": datetime(2000, 1, 1, tzinfo=timezone.utc),
    "ModDate": datetime(2000, 1, 1, tzinfo=timezone.utc),
}
PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
}


@dataclass(frozen=True)
class GroupResult:
    name: str
    mse: float
    interactions: int


@dataclass(frozen=True)
class GroupingResult:
    name: str
    rgrp: float
    groups: tuple[GroupResult, ...]


@dataclass(frozen=True)
class ModelResult:
    name: str
    rmse: float
    groupings: dict[str, GroupingResult]


@dataclass(frozen=True)
class DatasetResults:
    dataset: str
    grouping_order: tuple[str, ...]
    models: tuple[ModelResult, ...]


def generate_result_artifacts(
    input_path: Path | str,
    output_dir: Path | str | None = None,
) -> dict[str, Path]:
    """Read one evaluation JSON and create the table data and three PDF figures."""
    input_path = Path(input_path)
    results = _load_results(input_path)
    destination = (
        Path(output_dir)
        if output_dir is not None
        else input_path.parent / results.dataset
    )
    destination.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "boxplot": destination / "boxplot.pdf",
        "table": destination / "grp_unfairness_and_error_table.json",
        "unfairness": destination / "grp_unfairness_by_model_and_groups.pdf",
        "group_loss": destination / "grp_loss_by_model_and_groups.pdf",
    }

    _write_table(results, output_paths["table"])
    _save_figure(
        output_paths["unfairness"],
        lambda: _group_unfairness_figure(results),
    )
    _save_figure(
        output_paths["group_loss"],
        lambda: _group_loss_figure(results),
    )
    _save_figure(output_paths["boxplot"], lambda: _boxplot_figure(results))
    return output_paths


def _load_results(path: Path) -> DatasetResults:
    try:
        with path.open(encoding="utf-8") as input_file:
            document = json.load(input_file)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Results file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid results JSON: {path}") from error

    raw_models = document.get("results") if isinstance(document, dict) else None
    if not isinstance(raw_models, dict) or not raw_models:
        raise ValueError("Results JSON must contain a non-empty 'results' object")

    dataset_name = None
    grouping_order = None
    models = []
    for model_name in sorted(raw_models, key=_model_sort_key):
        raw_model = raw_models[model_name]
        if not isinstance(raw_model, dict):
            raise ValueError(f"Model {model_name} must be an object")

        evaluation = raw_model.get("evaluation")
        current_dataset = evaluation.get("dataset") if isinstance(evaluation, dict) else None
        if not isinstance(current_dataset, str) or not current_dataset:
            raise ValueError(f"Model {model_name} has no evaluation.dataset")
        if dataset_name is None:
            dataset_name = current_dataset
            try:
                grouping_order = _available_grouping_order(
                    current_dataset,
                    raw_models,
                )
            except KeyError as error:
                raise ValueError(
                    f"Unsupported results dataset: {current_dataset}"
                ) from error
        elif current_dataset != dataset_name:
            raise ValueError("All model entries must belong to the same dataset")

        raw_groupings = raw_model.get("groupings")
        if not isinstance(raw_groupings, dict):
            raise ValueError(f"Model {model_name} has no groupings object")
        missing = set(grouping_order) - set(raw_groupings)
        if missing:
            raise ValueError(
                f"Model {model_name} is missing groupings: {', '.join(sorted(missing))}"
            )

        groupings = {
            name: _parse_grouping(model_name, name, raw_groupings[name])
            for name in grouping_order
        }
        rmse = _model_rmse(model_name, groupings)
        models.append(ModelResult(model_name, rmse, groupings))

    return DatasetResults(dataset_name, grouping_order, tuple(models))


def _parse_grouping(
    model_name: str, grouping_name: str, raw_grouping: Any
) -> GroupingResult:
    if not isinstance(raw_grouping, dict):
        raise ValueError(f"{model_name}/{grouping_name} must be an object")
    if raw_grouping.get("valid") is not True:
        raise ValueError(f"{model_name}/{grouping_name} is not a valid partition")

    rgrp = _finite_nonnegative(
        raw_grouping.get("rgrp"), f"{model_name}/{grouping_name}.rgrp"
    )
    raw_groups = raw_grouping.get("groups")
    if not isinstance(raw_groups, dict) or not raw_groups:
        raise ValueError(f"{model_name}/{grouping_name} has no groups")

    groups = []
    for group_name in sorted(raw_groups, key=lambda name: _group_sort_key(grouping_name, name)):
        raw_group = raw_groups[group_name]
        if not isinstance(raw_group, dict):
            raise ValueError(
                f"{model_name}/{grouping_name}/{group_name} must be an object"
            )
        mse = _finite_nonnegative(
            raw_group.get("mse"),
            f"{model_name}/{grouping_name}/{group_name}.mse",
        )
        interactions = raw_group.get("test_interactions")
        if not isinstance(interactions, int) or interactions <= 0:
            raise ValueError(
                f"{model_name}/{grouping_name}/{group_name}.test_interactions "
                "must be a positive integer"
            )
        groups.append(GroupResult(group_name, mse, interactions))

    return GroupingResult(grouping_name, rgrp, tuple(groups))


def _model_rmse(
    model_name: str, groupings: dict[str, GroupingResult]
) -> float:
    mean_squared_errors = []
    for grouping in groupings.values():
        total_interactions = sum(group.interactions for group in grouping.groups)
        mse = sum(
            group.mse * group.interactions for group in grouping.groups
        ) / total_interactions
        mean_squared_errors.append(mse)

    reference = mean_squared_errors[0]
    for mse in mean_squared_errors[1:]:
        if not math.isclose(reference, mse, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(
                f"Model {model_name} has inconsistent global MSE across groupings"
            )
    return math.sqrt(reference)


def _write_table(results: DatasetResults, path: Path) -> None:
    rows = []
    for model in results.models:
        for grouping_name in results.grouping_order:
            rows.append(
                {
                    "model": model.name,
                    "grouping": GROUPING_LABELS[grouping_name],
                    "rgrp": model.groupings[grouping_name].rgrp,
                    "rmse": model.rmse,
                }
            )

    _atomic_json_write(
        path,
        {
            "dataset": results.dataset,
            "columns": ["model", "grouping", "rgrp", "rmse"],
            "rows": rows,
        },
    )


def _group_unfairness_figure(results: DatasetResults):
    model_count = len(results.models)
    with plt.rc_context(PLOT_STYLE):
        figure, axes = plt.subplots(
            1,
            model_count,
            figsize=(max(6.5, 3.8 * model_count), 4.0),
            sharey=True,
            squeeze=False,
        )
        for column, model in enumerate(results.models):
            axis = axes[0, column]
            values = [
                model.groupings[name].rgrp for name in results.grouping_order
            ]
            axis.bar(
                np.arange(len(results.grouping_order)),
                values,
                color=[
                    GROUPING_COLORS[name] for name in results.grouping_order
                ],
                width=0.72,
            )
            axis.set_title(model.name)
            axis.set_xticks([])
            axis.grid(axis="y", linestyle="--", alpha=0.35)
            axis.set_axisbelow(True)
            if column == 0:
                axis.set_ylabel("Group Unfairness")

        handles = [
            Patch(color=GROUPING_COLORS[name], label=GROUPING_LABELS[name])
            for name in results.grouping_order
        ]
        figure.legend(
            handles=handles,
            loc="lower center",
            ncol=len(results.grouping_order),
            frameon=True,
        )
        figure.subplots_adjust(bottom=0.22, left=0.1, right=0.98, top=0.9, wspace=0.18)
        return figure


def _group_loss_figure(results: DatasetResults):
    row_count = len(results.models)
    column_count = len(results.grouping_order)
    with plt.rc_context(PLOT_STYLE):
        figure, axes = plt.subplots(
            row_count,
            column_count,
            figsize=(3.35 * column_count, 2.8 * row_count),
            sharey=True,
            squeeze=False,
        )
        for row, model in enumerate(results.models):
            for column, grouping_name in enumerate(results.grouping_order):
                axis = axes[row, column]
                grouping = model.groupings[grouping_name]
                positions = np.arange(len(grouping.groups))
                axis.bar(
                    positions,
                    [group.mse for group in grouping.groups],
                    color=_color_gradient(
                        GROUPING_COLORS[grouping_name], len(grouping.groups)
                    ),
                    width=0.72,
                )
                axis.set_title(f"{model.name} – {GROUPING_LABELS[grouping_name]}")
                axis.set_xticks(
                    positions,
                    [_group_label(group.name) for group in grouping.groups],
                    rotation=30 if len(grouping.groups) > 5 else 0,
                    ha="right" if len(grouping.groups) > 5 else "center",
                )
                axis.grid(axis="y", linestyle="--", alpha=0.3)
                axis.set_axisbelow(True)
                if column == 0:
                    axis.set_ylabel("Group Loss (MSE)")

        figure.tight_layout(pad=1.1)
        return figure


def _boxplot_figure(results: DatasetResults):
    distributions = [
        [model.groupings[name].rgrp for name in results.grouping_order]
        for model in results.models
    ]
    with plt.rc_context(PLOT_STYLE):
        figure, axis = plt.subplots(
            figsize=(max(6.5, 1.6 * len(results.models) + 3.5), 4.5)
        )
        axis.boxplot(
            distributions,
            tick_labels=[model.name for model in results.models],
            patch_artist=True,
            showmeans=True,
            boxprops={"facecolor": "#9bd3e5", "edgecolor": "#2455ff"},
            whiskerprops={"color": "#555555"},
            capprops={"color": "#555555"},
            medianprops={"color": "#ff7f0e", "linewidth": 1.5},
            meanprops={
                "marker": "s",
                "markerfacecolor": "green",
                "markeredgecolor": "green",
                "markersize": 7,
            },
        )
        axis.set_xlabel("Recommendation Model")
        axis.set_ylabel("Group Unfairness")
        axis.grid(axis="y", linestyle="--", alpha=0.4)
        axis.set_axisbelow(True)
        figure.tight_layout()
        return figure


def _save_figure(path: Path, create_figure: Callable[[], Any]) -> None:
    temporary_path = None
    figure = None
    try:
        figure = create_figure()
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        figure.savefig(
            temporary_path,
            format="pdf",
            bbox_inches="tight",
            metadata=PDF_METADATA,
        )
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise
    finally:
        if figure is not None:
            plt.close(figure)


def _atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(value, temporary_file, indent=2, allow_nan=False)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise


def _model_sort_key(model_name: str):
    return (MODEL_PRIORITY.get(model_name, len(MODEL_PRIORITY)), model_name.casefold())


def _available_grouping_order(dataset_name: str, raw_models: dict[str, Any]):
    grouping_order = GROUPING_ORDER_BY_DATASET[dataset_name.casefold()]
    if dataset_name.casefold() != "yelp":
        return grouping_order

    has_fans = all(
        isinstance(model, dict)
        and isinstance(model.get("groupings"), dict)
        and "fans" in model["groupings"]
        for model in raw_models.values()
    )
    if has_fans:
        return grouping_order

    return tuple(name for name in grouping_order if name != "fans")


def _group_sort_key(grouping_name: str, group_name: str):
    predefined = {
        "activity": {"active": 0, "inactive": 1},
        "age": {
            "under_18": 0,
            "18_24": 1,
            "25_34": 2,
            "35_44": 3,
            "45_49": 4,
            "50_55": 5,
            "over_55": 6,
            "unknown": 7,
        },
        "gender": {"male": 0, "female": 1, "unknown": 2},
        "friend_count": {
            "no_friends": 0,
            "1_10": 1,
            "11_100": 2,
            "101_plus": 3,
        },
        "fans": {
            "no_fans": 0,
            "1_10": 1,
            "11_100": 2,
            "101_plus": 3,
        },
        "tenure": {
            "under_1": 0,
            "1_3": 1,
            "3_5": 2,
            "5_plus": 3,
        },
    }
    if grouping_name in predefined:
        order = predefined[grouping_name]
        return (order.get(group_name, len(order)), group_name)
    if group_name.startswith("group_"):
        suffix = group_name.removeprefix("group_")
        if suffix.isdigit():
            return (0, int(suffix))
    return (1, group_name)


def _group_label(group_name: str) -> str:
    labels = {
        "active": "Active",
        "inactive": "Inactive",
        "male": "Male",
        "female": "Female",
        "unknown": "Unknown",
        "under_18": "<18",
        "18_24": "18–24",
        "25_34": "25–34",
        "35_44": "35–44",
        "45_49": "45–49",
        "50_55": "50–55",
        "over_55": ">55",
        "no_friends": "None",
        "no_fans": "None",
        "1_10": "1–10",
        "11_100": "11–100",
        "101_plus": "101+",
        "under_1": "<1",
        "1_3": "1–3",
        "3_5": "3–5",
        "5_plus": "5+",
    }
    if group_name in labels:
        return labels[group_name]
    if group_name.startswith("group_"):
        suffix = group_name.removeprefix("group_")
        if suffix.isdigit():
            return f"G{int(suffix)}"
    return group_name.replace("_", " ").title()


def _color_gradient(color: str, size: int) -> list[tuple[float, float, float]]:
    base = np.asarray(matplotlib.colors.to_rgb(color))
    strengths = np.linspace(0.35, 1.0, size) if size > 1 else np.asarray([0.75])
    return [tuple(1.0 - strength * (1.0 - base)) for strength in strengths]


def _finite_nonnegative(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate publication figures and table data from results JSON."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. lastfm.")
    parser.add_argument(
        "--input",
        type=Path,
        help="Input JSON; defaults to results/results_<dataset>.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory; defaults to results/<dataset>.",
    )
    return parser


def main() -> None:
    arguments = _build_parser().parse_args()
    input_path = arguments.input or (
        REPOSITORY_ROOT / "results" / f"results_{arguments.dataset}.json"
    )
    output_dir = arguments.output_dir or (
        REPOSITORY_ROOT / "results" / arguments.dataset
    )
    paths = generate_result_artifacts(input_path, output_dir)
    for name, path in paths.items():
        styled_print(f"{name}: {path}", ConsoleColor.YELLOW)


if __name__ == "__main__":
    main()
