import csv
from collections import Counter
from pathlib import Path
from typing import Callable


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class LastFMSampler:
    """Create a small, dense LastFM sample in RecBole's atomic format."""

    def __init__(
        self,
        source_dir: Path | str = REPOSITORY_ROOT / "data/processed/lastfm",
        output_dir: Path | str = REPOSITORY_ROOT / "data/sample/lastfm",
        user_limit: int = 20,
        item_limit: int = 20,
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
        ):
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
        interaction_count = self._write_filtered_file(
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
