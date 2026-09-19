import csv
import fcntl
import json
import math
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "recsys-fairness-matplotlib")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from src.fairness.grouping import Partition, UserProfile, metadata_partitions

LATENT_GROUPING_NAMES = ("kmeans", "agglomerative")

GROUPING_CONFIG = {
    "lastfm": {
        "activity": {
            "order": ("active", "inactive"),
            "labels": {"active": "Active", "inactive": "Inactive"},
            "title": "User Distribution by Activity",
            "xlabel": "Activity",
        },
        "age": {
            "order": (
                "under_18",
                "18_24",
                "25_34",
                "35_44",
                "45_49",
                "50_55",
                "over_55",
                "unknown",
            ),
            "labels": {
                "under_18": "Under 18",
                "18_24": "18–24",
                "25_34": "25–34",
                "35_44": "35–44",
                "45_49": "45–49",
                "50_55": "50–55",
                "over_55": "Over 55",
                "unknown": "Unknown",
            },
            "title": "User Distribution by Age",
            "xlabel": "Age Group",
        },
        "gender": {
            "order": ("male", "female", "unknown"),
            "labels": {
                "male": "Male",
                "female": "Female",
                "unknown": "Unknown",
            },
            "title": "User Distribution by Gender",
            "xlabel": "Gender",
        },
        "location": {
            "order": ("north_america", "europe", "other_locations"),
            "labels": {
                "north_america": "North America",
                "europe": "Europe",
                "other_locations": "Other Locations",
            },
            "title": "User Distribution by Location",
            "xlabel": "Location",
        },
    },
    "yelp": {
        "activity": {
            "order": ("active", "inactive"),
            "labels": {"active": "Active", "inactive": "Inactive"},
            "title": "User Distribution by Activity",
            "xlabel": "Activity",
        },
        "friend_count": {
            "order": ("no_friends", "1_10", "11_100", "101_plus"),
            "labels": {
                "no_friends": "No friends",
                "1_10": "1–10",
                "11_100": "11–100",
                "101_plus": "101+",
            },
            "title": "User Distribution by Friend Count",
            "xlabel": "Number of Friends",
        },
        "fans": {
            "order": ("0_1", "2_7", "8_13", "14_plus"),
            "labels": {
                "0_1": "0–1",
                "2_7": "2–7",
                "8_13": "8–13",
                "14_plus": "14+",
            },
            "title": "User Distribution by Fan Count",
            "xlabel": "Number of Fans",
        },
        "tenure": {
            "order": ("under_1", "1_3", "3_5", "5_plus"),
            "labels": {
                "under_1": "< 1 year",
                "1_3": "1–3 years",
                "3_5": "3–5 years",
                "5_plus": "5+ years",
            },
            "title": "User Distribution by Tenure",
            "xlabel": "Tenure",
        },
    },
}

PDF_METADATA = {
    "Creator": "recsys_fairness",
    "Producer": "Matplotlib",
    "CreationDate": datetime(2000, 1, 1, tzinfo=timezone.utc),
    "ModDate": datetime(2000, 1, 1, tzinfo=timezone.utc),
}


def generate_sample_statistics(sample_dir, dataset, output_dir):
    """
    Generate statistics and user group distribution plots for a sampled dataset.

    Args:
        sample_dir: Directory containing the sampled atomic files.
        dataset: Dataset name.
        output_dir: Directory where statistics and plots will be written.

    Returns:
        paths: Paths of the generated artifacts.
    """
    sample_dir = Path(sample_dir)
    output_dir = Path(output_dir)
    dataset = dataset.casefold()

    if dataset not in GROUPING_CONFIG:
        raise ValueError(f"Unsupported statistics dataset: {dataset}")

    profiles = _read_profiles(sample_dir / f"{dataset}.user", dataset)
    users = len(profiles)
    items = _count_data_rows(sample_dir / f"{dataset}.item")
    interactions = _count_data_rows(sample_dir / f"{dataset}.inter")

    if users == 0 or items == 0:
        raise ValueError("Cannot compute density for an empty sample")

    density = interactions / (users * items) * 100
    partitions = metadata_partitions(profiles, dataset)
    groupings = {}

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"statistics": output_dir / "statistics.json"}

    for grouping_name, config in GROUPING_CONFIG[dataset].items():
        partition = partitions[grouping_name]
        counts = Counter(partition.assignments.values())
        unexpected_groups = set(counts) - set(config["order"])
        if unexpected_groups:
            raise ValueError(
                f"Unexpected {grouping_name} groups: "
                f"{', '.join(sorted(unexpected_groups))}"
            )

        groupings[grouping_name] = {
            group_name: {
                "users": counts[group_name],
                "percentage": f"{counts[group_name] / users * 100:.2f}%",
            }
            for group_name in config["order"]
        }

        output_path = output_dir / f"{grouping_name}_distribution.pdf"
        _save_group_distribution(
            output_path,
            counts,
            users,
            config,
        )
        paths[grouping_name] = output_path

    statistics = {
        "users": users,
        "items": items,
        "interactions": interactions,
        "density": f"{density:.6f}%",
        "groupings": groupings,
    }
    _atomic_json_write(paths["statistics"], statistics)

    return paths


def persist_latent_group_statistics(output_dir, partitions):
    """Add final-test latent group statistics to the sample artifacts."""
    group_counts = {}
    clustering = {}
    population_size = None

    for grouping_name in LATENT_GROUPING_NAMES:
        try:
            partition = partitions[grouping_name]
        except KeyError as error:
            raise ValueError(
                f"Missing latent partition: {grouping_name}"
            ) from error

        if not isinstance(partition, Partition):
            raise TypeError(f"{grouping_name} must be a Partition")

        current_population_size = len(partition.assignments)
        if current_population_size == 0:
            raise ValueError(f"{grouping_name} partition is empty")
        if population_size is None:
            population_size = current_population_size
        elif current_population_size != population_size:
            raise ValueError("Latent partitions have different population sizes")

        metadata = partition.metadata
        if metadata.get("type") != "latent":
            raise ValueError(f"{grouping_name} is not a latent partition")

        group_counts[grouping_name] = Counter(partition.assignments.values())
        clustering[grouping_name] = metadata

    return persist_latent_group_summaries(
        output_dir=output_dir,
        group_counts=group_counts,
        clustering=clustering,
        population_size=population_size,
    )


def persist_latent_group_summaries(
    output_dir,
    group_counts,
    clustering,
    population_size,
):
    """Persist validated latent group counts and clustering metadata."""
    output_dir = Path(output_dir)
    statistics_path = output_dir / "statistics.json"
    lock_path = output_dir / ".statistics.json.lock"

    if isinstance(population_size, bool) or not isinstance(population_size, int):
        raise TypeError("population_size must be an integer")
    if population_size <= 0:
        raise ValueError("population_size must be positive")

    latent_groupings = {}
    normalized_clustering = {}
    plot_data = {}

    for grouping_name in LATENT_GROUPING_NAMES:
        try:
            raw_counts = group_counts[grouping_name]
            metadata = clustering[grouping_name]
        except KeyError as error:
            raise ValueError(
                f"Missing latent grouping summary: {grouping_name}"
            ) from error

        if not isinstance(raw_counts, dict):
            raise ValueError(f"{grouping_name} counts must be an object")
        if not isinstance(metadata, dict):
            raise ValueError(f"{grouping_name} metadata must be an object")

        try:
            selected_k = int(metadata["selected_k"])
            selected_silhouette = float(metadata["selected_silhouette"])
            raw_silhouettes = metadata["silhouette_by_k"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid clustering metadata for {grouping_name}"
            ) from error

        if selected_k < 2 or not math.isfinite(selected_silhouette):
            raise ValueError(f"Invalid selected clustering for {grouping_name}")
        if not isinstance(raw_silhouettes, dict):
            raise ValueError(
                f"{grouping_name}.silhouette_by_k must be an object"
            )

        silhouettes = {}
        for raw_k, raw_score in raw_silhouettes.items():
            try:
                k = int(raw_k)
                score = float(raw_score)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid silhouette entry for {grouping_name}"
                ) from error
            if k < 2 or not math.isfinite(score):
                raise ValueError(
                    f"Invalid silhouette entry for {grouping_name}: k={raw_k}"
                )
            silhouettes[str(k)] = score

        if str(selected_k) not in silhouettes:
            raise ValueError(
                f"Selected k is missing from {grouping_name}.silhouette_by_k"
            )

        order = tuple(f"group_{index}" for index in range(1, selected_k + 1))
        if set(raw_counts) != set(order):
            raise ValueError(
                f"Unexpected groups in {grouping_name}: "
                f"{', '.join(sorted(raw_counts))}"
            )

        counts = Counter()
        for group_name in order:
            users = raw_counts[group_name]
            if isinstance(users, bool) or not isinstance(users, int) or users <= 0:
                raise ValueError(
                    f"{grouping_name}/{group_name} users must be a positive integer"
                )
            counts[group_name] = users

        if sum(counts.values()) != population_size:
            raise ValueError(
                f"{grouping_name} counts do not total {population_size} users"
            )

        latent_groupings[grouping_name] = {
            group_name: {
                "users": counts[group_name],
                "percentage": f"{counts[group_name] / population_size * 100:.2f}%",
            }
            for group_name in order
        }
        normalized_clustering[grouping_name] = {
            "selected_k": selected_k,
            "selected_silhouette": selected_silhouette,
            "silhouette_by_k": {
                key: silhouettes[key]
                for key in sorted(silhouettes, key=int)
            },
        }
        plot_data[grouping_name] = (
            counts,
            {
                "order": order,
                "labels": {
                    group_name: f"Group {index}"
                    for index, group_name in enumerate(order, start=1)
                },
                "title": (
                    "User Distribution by "
                    f"{'K-Means' if grouping_name == 'kmeans' else 'Agglomerative'} "
                    "Cluster"
                ),
                "xlabel": "Cluster",
            },
        )

    if not statistics_path.exists():
        raise FileNotFoundError(
            f"Sample statistics file not found: {statistics_path}"
        )

    paths = {"statistics": statistics_path}
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        with statistics_path.open(encoding="utf-8") as input_file:
            statistics = json.load(input_file)

        sample_users = statistics.get("users")
        if sample_users != population_size:
            raise ValueError(
                "Latent partition population does not match sample statistics: "
                f"{population_size} != {sample_users}"
            )

        groupings = statistics.get("groupings")
        if not isinstance(groupings, dict):
            raise ValueError("Sample statistics has no groupings object")

        existing_clustering = statistics.get("clustering", {})
        if not isinstance(existing_clustering, dict):
            raise ValueError("Sample statistics clustering must be an object")

        for grouping_name in LATENT_GROUPING_NAMES:
            if (
                grouping_name in groupings
                and groupings[grouping_name] != latent_groupings[grouping_name]
            ):
                raise ValueError(
                    f"Conflicting {grouping_name} distribution already persisted"
                )
            if (
                grouping_name in existing_clustering
                and existing_clustering[grouping_name]
                != normalized_clustering[grouping_name]
            ):
                raise ValueError(
                    f"Conflicting {grouping_name} metadata already persisted"
                )

        for grouping_name in LATENT_GROUPING_NAMES:
            counts, config = plot_data[grouping_name]
            output_path = output_dir / f"{grouping_name}_distribution.pdf"
            _save_group_distribution(
                output_path,
                counts,
                population_size,
                config,
            )
            paths[grouping_name] = output_path
            groupings[grouping_name] = latent_groupings[grouping_name]
            existing_clustering[grouping_name] = normalized_clustering[
                grouping_name
            ]

        statistics["clustering"] = existing_clustering
        _atomic_json_write(statistics_path, statistics)

    return paths


def _read_profiles(path, dataset):
    profiles = []

    with path.open(encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file, delimiter="\t")

        for row in reader:
            if dataset == "lastfm":
                raw_age = row["age:float"].strip()
                profiles.append(
                    UserProfile(
                        user_id=row["user_id:token"],
                        development_interactions=0,
                        gender=row["gender:token"],
                        age=float(raw_age) if raw_age else None,
                        country=row["country:token"],
                        is_active=(
                            row["is_active:token"].strip().casefold() == "true"
                        ),
                    )
                )
            else:
                profiles.append(
                    UserProfile(
                        user_id=row["user_id:token"],
                        development_interactions=0,
                        is_active=(
                            row["is_active:token"].strip().casefold() == "true"
                        ),
                        friend_count=float(row["friend_count:float"]),
                        fans=float(row["fans:float"]),
                        tenure_years=float(row["tenure_years:float"]),
                    )
                )

    return profiles


def _save_group_distribution(path, counts, total, config):
    order = config["order"]
    percentages = [counts[group] / total * 100 for group in order]
    labels = [config["labels"][group] for group in order]

    figure, axis = plt.subplots(figsize=(8, 5.5))
    bars = axis.bar(
        labels,
        percentages,
        width=0.65,
        edgecolor="black",
        linewidth=0.7,
    )

    for bar, group, percentage in zip(bars, order, percentages):
        axis.annotate(
            f"{percentage:.2f}%\n({counts[group]:,})",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    axis.set_title(config["title"], fontsize=15, pad=14)
    axis.set_xlabel(config["xlabel"], fontsize=11)
    axis.set_ylabel("Percentage of Users", fontsize=11)
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    axis.set_ylim(0, max(percentages) * 1.18)
    axis.grid(axis="y", linestyle="--", alpha=0.25)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.text(
        0.99,
        0.98,
        f"N = {total:,} users",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        alpha=0.65,
    )
    figure.tight_layout()

    temporary_path = None
    try:
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
        plt.close(figure)


def _atomic_json_write(path, value):
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
            json.dump(value, output_file, indent=2, allow_nan=False)
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())

        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise


def _count_data_rows(path):
    line_count = 0
    last_byte = b""

    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            line_count += chunk.count(b"\n")
            last_byte = chunk[-1:]

    if last_byte and last_byte != b"\n":
        line_count += 1

    return max(line_count - 1, 0)
