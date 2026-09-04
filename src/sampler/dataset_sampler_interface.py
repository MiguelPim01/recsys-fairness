import csv
from collections import Counter
from pathlib import Path

from src.utils.console import styled_tqdm


class IDatasetSampler:
    """Create a deterministic user-item sample from RecBole atomic files."""

    DATASET_NAME = None

    def __init__(
        self,
        source_dir,
        output_dir,
        user_limit=50,
        item_limit=50,
        seed=42,
        minimum_user_interactions=6,
    ):
        if user_limit <= 0 or item_limit <= 0:
            raise ValueError("user_limit and item_limit must be positive")
        if minimum_user_interactions <= 0:
            raise ValueError("minimum_user_interactions must be positive")

        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.user_limit = user_limit
        self.item_limit = item_limit
        self.seed = seed
        self.minimum_user_interactions = minimum_user_interactions

    def create_sample(self):
        """Select the most popular items and the most active eligible users."""
        interaction_path = self.source_dir / f"{self.DATASET_NAME}.inter"
        user_path = self.source_dir / f"{self.DATASET_NAME}.user"
        item_path = self.source_dir / f"{self.DATASET_NAME}.item"
        self._require_files(interaction_path, user_path, item_path)

        interaction_total = self._count_data_rows(interaction_path)
        user_total = self._count_data_rows(user_path)
        item_total = self._count_data_rows(item_path)

        selected_items, source_interactions = self._select_items(
            interaction_path,
            interaction_total,
        )
        eligible_users = self._eligible_users(
            interaction_path,
            selected_items,
            interaction_total,
        )
        selected_users = self._select_users(eligible_users)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        output_interaction_path = self.output_dir / f"{self.DATASET_NAME}.inter"
        output_user_path = self.output_dir / f"{self.DATASET_NAME}.user"
        output_item_path = self.output_dir / f"{self.DATASET_NAME}.item"

        interaction_count, interacted_items = self._write_interactions(
            interaction_path,
            output_interaction_path,
            selected_users,
            selected_items,
            interaction_total,
        )
        written_users = self._write_selected_entities(
            user_path,
            output_user_path,
            selected_users,
            "users",
            user_total,
        )
        written_items = self._write_selected_entities(
            item_path,
            output_item_path,
            selected_items,
            "items",
            item_total,
        )

        if written_users != selected_users:
            missing_users = selected_users - written_users
            raise ValueError(
                f"Missing profiles for {len(missing_users)} sampled users"
            )
        if written_items != selected_items:
            missing_items = selected_items - written_items
            raise ValueError(
                f"Missing profiles for {len(missing_items)} sampled items"
            )

        matrix_size = len(selected_users) * len(selected_items)
        density = interaction_count / matrix_size * 100

        return {
            "source_interactions": source_interactions,
            "eligible_users": len(eligible_users),
            "selected_users": len(selected_users),
            "selected_items": len(selected_items),
            "sample_interactions": interaction_count,
            "density": f"{density:.4f}%",
            "items_without_interactions": len(selected_items - interacted_items),
            "seed": self.seed,
            "minimum_user_interactions": self.minimum_user_interactions,
        }

    def _select_items(self, interaction_path, interaction_total):
        item_interactions = Counter()
        interaction_count = 0

        with interaction_path.open(encoding="utf-8", newline="") as input_file:
            reader = csv.reader(input_file, delimiter="\t")
            self._require_header(reader, interaction_path)

            progress = styled_tqdm(
                reader,
                total=interaction_total,
                desc="  ranking items",
                unit="interaction",
                dynamic_ncols=True,
            )
            for row in progress:
                self._validate_row(row, interaction_path, minimum_columns=2)
                item_interactions[row[1]] += 1
                interaction_count += 1

        if len(item_interactions) < self.item_limit:
            raise ValueError(
                f"Dataset contains only {len(item_interactions)} items; "
                f"{self.item_limit} are required"
            )

        ranked_items = sorted(
            item_interactions,
            key=lambda item_id: (-item_interactions[item_id], item_id),
        )

        return set(ranked_items[:self.item_limit]), interaction_count

    def _eligible_users(self, interaction_path, selected_items, interaction_total):
        """
        Finds users with enough interactions among the selected items.

        Args:
            interaction_path: Path to the RecBole interaction file.
            selected_items: Set containing the selected item IDs.
            interaction_total: Total number of interactions in the file.

        Returns:
            Dictionary mapping eligible user IDs to their global interaction counts.
        """
        global_user_interactions = Counter()
        sample_user_interactions = Counter()

        with interaction_path.open(encoding="utf-8", newline="") as input_file:
            reader = csv.reader(input_file, delimiter="\t")
            self._require_header(reader, interaction_path)

            progress = styled_tqdm(
                reader,
                total=interaction_total,
                desc="  finding users",
                unit="interaction",
                dynamic_ncols=True,
            )
            for row in progress:
                self._validate_row(row, interaction_path, minimum_columns=2)
                global_user_interactions[row[0]] += 1

                if row[1] in selected_items:
                    sample_user_interactions[row[0]] += 1

        return {
            user_id: global_user_interactions[user_id]
            for user_id, interaction_count in sample_user_interactions.items()
            if interaction_count >= self.minimum_user_interactions
        }

    def _select_users(self, eligible_users):
        """
        Selects the most active eligible users.

        Args:
            eligible_users: Dictionary mapping user IDs to interaction counts.

        Returns:
            Set containing the selected user IDs.
        """
        if len(eligible_users) < self.user_limit:
            raise ValueError(
                f"Dataset contains only {len(eligible_users)} eligible users; "
                f"{self.user_limit} are required"
            )

        ranked_users = sorted(
            eligible_users,
            key=lambda user_id: (-eligible_users[user_id], user_id),
        )

        return set(ranked_users[:self.user_limit])

    def _write_interactions(
        self,
        source_path,
        output_path,
        selected_users,
        selected_items,
        interaction_total,
    ):
        interaction_count = 0
        interacted_items = set()

        with source_path.open(
            encoding="utf-8",
            newline="",
        ) as input_file, output_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as output_file:
            reader = csv.reader(input_file, delimiter="\t")
            writer = csv.writer(output_file, delimiter="\t", lineterminator="\n")
            writer.writerow(self._require_header(reader, source_path))

            progress = styled_tqdm(
                reader,
                total=interaction_total,
                desc="  interactions",
                unit="interaction",
                dynamic_ncols=True,
            )
            for row in progress:
                self._validate_row(row, source_path, minimum_columns=2)
                if row[0] not in selected_users or row[1] not in selected_items:
                    continue

                writer.writerow(row)
                interacted_items.add(row[1])
                interaction_count += 1

        if not interaction_count:
            raise ValueError(f"The {self.DATASET_NAME} sample has no interactions")

        return interaction_count, interacted_items

    @staticmethod
    def _write_selected_entities(
        source_path,
        output_path,
        selected_ids,
        description,
        entity_total,
    ):
        written_ids = set()

        with source_path.open(
            encoding="utf-8",
            newline="",
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

            progress = styled_tqdm(
                reader,
                total=entity_total,
                desc=f"  {description}",
                unit=description[:-1],
                dynamic_ncols=True,
            )
            for row in progress:
                IDatasetSampler._validate_row(
                    row,
                    source_path,
                    minimum_columns=1,
                )
                if row[0] in selected_ids:
                    writer.writerow(row)
                    written_ids.add(row[0])

        return written_ids

    @staticmethod
    def _require_header(reader, path):
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Atomic file is empty: {path}")

        return header

    @staticmethod
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

    @staticmethod
    def _validate_row(row, path, minimum_columns):
        if len(row) < minimum_columns:
            raise ValueError(
                f"Malformed row in {path}: expected at least "
                f"{minimum_columns} columns"
            )

    @staticmethod
    def _require_files(*paths):
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing atomic files: {', '.join(missing)}"
            )
