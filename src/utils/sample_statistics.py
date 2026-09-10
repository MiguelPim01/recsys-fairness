import csv
import json
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

from src.fairness.grouping import UserProfile, metadata_partitions

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
