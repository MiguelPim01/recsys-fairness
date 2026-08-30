import csv
import math
from pathlib import Path

from tqdm.auto import tqdm

from src.sampler.dataset_sampler_interface import IDatasetSampler

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class LastFMSampler(IDatasetSampler):
    """Create a reproducible LastFM sample in RecBole's atomic format."""

    DATASET_NAME = "lastfm"

    def __init__(
        self,
        source_dir=REPOSITORY_ROOT / "data/processed/lastfm",
        output_dir=REPOSITORY_ROOT / "data/sample/lastfm",
        user_limit=1000,
        item_limit=1000,
        seed=42,
        minimum_user_interactions=6,
    ):
        super().__init__(
            source_dir,
            output_dir,
            user_limit,
            item_limit,
            seed,
            minimum_user_interactions,
        )

    def _write_interactions(
        self,
        source_path,
        output_path,
        selected_users,
        selected_items,
    ):
        interactions = []
        interacted_items = set()

        with source_path.open(encoding="utf-8", newline="") as input_file:
            reader = csv.reader(input_file, delimiter="\t")
            self._require_header(reader, source_path)

            progress = tqdm(
                reader,
                desc="  interactions",
                unit="interaction",
                dynamic_ncols=True,
            )
            for row in progress:
                self._validate_row(row, source_path, minimum_columns=3)
                if row[0] not in selected_users or row[1] not in selected_items:
                    continue

                try:
                    play_count = float(row[2])
                except ValueError as error:
                    raise ValueError(
                        f"Invalid play count in {source_path}: {row[2]}"
                    ) from error

                if not math.isfinite(play_count) or play_count <= 0:
                    raise ValueError(
                        f"Play count must be finite and positive: {play_count}"
                    )

                interactions.append((row[0], row[1], play_count))
                interacted_items.add(row[1])

        if not interactions:
            raise ValueError("The lastfm sample has no interactions")

        logged_counts = [math.log1p(row[2]) for row in interactions]
        minimum = min(logged_counts)
        maximum = max(logged_counts)

        with output_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.writer(output_file, delimiter="\t", lineterminator="\n")
            writer.writerow(["user_id:token", "item_id:token", "rating:float"])

            for (user_id, item_id, _), logged_count in zip(
                interactions,
                logged_counts,
            ):
                if math.isclose(minimum, maximum):
                    rating = 3.0
                else:
                    rating = 1.0 + 4.0 * (logged_count - minimum) / (
                        maximum - minimum
                    )

                writer.writerow([user_id, item_id, rating])

        return len(interactions), interacted_items
