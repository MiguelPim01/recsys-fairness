import csv
import math
from collections import Counter
from collections.abc import Callable
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class LastFMSampler:
    """Create a small, dense LastFM sample in RecBole's atomic format."""

    def __init__(
        self,
        source_dir: Path | str = REPOSITORY_ROOT / "data/processed/lastfm",
        output_dir: Path | str = REPOSITORY_ROOT / "data/sample/lastfm",
        user_limit: int = 50,
        item_limit: int = 50,
    ) -> None:
        if user_limit <= 0 or item_limit <= 0:
            raise ValueError("user_limit and item_limit must be positive")

        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.user_limit = user_limit
        self.item_limit = item_limit

    def create_sample(self) -> dict[str, int]:
        """Select a dense user-item subset and persist its atomic files."""
        output_interaction_path = self.output_dir / "lastfm.inter"
        output_user_path = self.output_dir / "lastfm.user"
        output_item_path = self.output_dir / "lastfm.item"
        if all(
            path.is_file()
            for path in (
                output_interaction_path,
                output_user_path,
                output_item_path,
            )
        ) and self._is_normalized_interaction_file(output_interaction_path):
            return {
                "users": self._count_data_rows(output_user_path),
                "items": self._count_data_rows(output_item_path),
                "interactions": self._count_data_rows(output_interaction_path),
            }

        interaction_path = self.source_dir / "lastfm.inter"
        user_path = self.source_dir / "lastfm.user"
        item_path = self.source_dir / "lastfm.item"
        self._require_files(interaction_path, user_path, item_path)

        selected_items = self._select_items(interaction_path)
        selected_users = self._select_users(interaction_path, selected_items)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        interaction_count = self._write_normalized_interactions(
            interaction_path,
            output_interaction_path,
            lambda row: row[0] in selected_users and row[1] in selected_items,
        )
        user_count = self._write_filtered_file(
            user_path,
            output_user_path,
            lambda row: row[0] in selected_users,
        )
        item_count = self._write_filtered_file(
            item_path,
            output_item_path,
            lambda row: row[0] in selected_items,
        )

        if user_count != self.user_limit:
            raise ValueError(
                f"Expected {self.user_limit} user records, found {user_count}"
            )
        if item_count != self.item_limit:
            raise ValueError(
                f"Expected {self.item_limit} item records, found {item_count}"
            )

        return {
            "users": user_count,
            "items": item_count,
            "interactions": interaction_count,
        }

    def _write_normalized_interactions(
        self,
        source_path: Path,
        output_path: Path,
        include_row: Callable[[list[str]], bool],
    ) -> int:
        interactions = []

        with source_path.open(encoding="utf-8", newline="") as input_file:
            reader = csv.reader(input_file, delimiter="\t")
            next(reader, None)

            for row in reader:
                self._validate_row(row, source_path, minimum_columns=3)
                if not include_row(row):
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

        if not interactions:
            raise ValueError("The LastFM sample contains no interactions")

        logged_counts = [math.log1p(row[2]) for row in interactions]
        minimum = min(logged_counts)
        maximum = max(logged_counts)

        with output_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.writer(output_file, delimiter="\t", lineterminator="\n")
            writer.writerow(["user_id:token", "item_id:token", "rating:float"])

            for (user_id, item_id, _), logged_count in zip(
                interactions, logged_counts
            ):
                if math.isclose(minimum, maximum):
                    rating = 3.0
                else:
                    rating = 1.0 + 4.0 * (logged_count - minimum) / (
                        maximum - minimum
                    )
                writer.writerow([user_id, item_id, rating])

        return len(interactions)

    def _select_items(self, interaction_path: Path) -> set[str]:
        item_interactions: Counter[str] = Counter()
        with interaction_path.open(encoding="utf-8", newline="") as input_file:
            reader = csv.reader(input_file, delimiter="\t")
            next(reader, None)
            for row in reader:
                self._validate_row(row, interaction_path, minimum_columns=2)
                item_interactions[row[1]] += 1

        return set(
            self._most_frequent_ids(
                item_interactions, self.item_limit, entity_name="items"
            )
        )

    def _select_users(
        self, interaction_path: Path, selected_items: set[str]
    ) -> set[str]:
        user_interactions: Counter[str] = Counter()
        with interaction_path.open(encoding="utf-8", newline="") as input_file:
            reader = csv.reader(input_file, delimiter="\t")
            next(reader, None)
            for row in reader:
                self._validate_row(row, interaction_path, minimum_columns=2)
                if row[1] in selected_items:
                    user_interactions[row[0]] += 1

        return set(
            self._most_frequent_ids(
                user_interactions, self.user_limit, entity_name="users"
            )
        )

    @staticmethod
    def _most_frequent_ids(
        frequencies: Counter[str], limit: int, entity_name: str
    ) -> list[str]:
        if len(frequencies) < limit:
            raise ValueError(
                f"Dataset contains only {len(frequencies)} eligible {entity_name}; "
                f"{limit} are required"
            )
        ranked = sorted(
            frequencies,
            key=lambda identifier: (-frequencies[identifier], identifier),
        )
        return ranked[:limit]

    @staticmethod
    def _write_filtered_file(
        source_path: Path,
        output_path: Path,
        include_row: Callable[[list[str]], bool],
    ) -> int:
        written_rows = 0
        with source_path.open(
            encoding="utf-8", newline=""
        ) as input_file, output_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as output_file:
            reader = csv.reader(input_file, delimiter="\t")
            writer = csv.writer(output_file, delimiter="\t", lineterminator="\n")
            header = next(reader, None)
            if header is None:
                raise ValueError(f"Atomic file is empty: {source_path}")
            writer.writerow(header)

            for row in reader:
                if include_row(row):
                    writer.writerow(row)
                    written_rows += 1

        return written_rows

    @staticmethod
    def _count_data_rows(path: Path) -> int:
        with path.open(encoding="utf-8", newline="") as input_file:
            reader = csv.reader(input_file, delimiter="\t")
            next(reader, None)
            return sum(1 for _ in reader)

    @staticmethod
    def _is_normalized_interaction_file(path: Path) -> bool:
        try:
            with path.open(encoding="utf-8", newline="") as input_file:
                header = next(csv.reader(input_file, delimiter="\t"), None)
        except OSError:
            return False

        return header is not None and len(header) >= 3 and header[2] == "rating:float"

    @staticmethod
    def _validate_row(row: list[str], path: Path, minimum_columns: int) -> None:
        if len(row) < minimum_columns:
            raise ValueError(
                f"Malformed row in {path}: expected at least {minimum_columns} columns"
            )

    @staticmethod
    def _require_files(*paths: Path) -> None:
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing LastFM atomic files: {', '.join(missing)}")
