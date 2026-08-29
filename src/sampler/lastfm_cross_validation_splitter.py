import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path


class LastFMCrossValidationSplitter:
    """Create reusable per-user folds and an isolated final test set."""

    MANIFEST_FILENAME = "lastfm.cv_manifest.json"

    def __init__(self, dataset_dir, n_splits = 5, seed = 42, test_ratio = 0.2):
        self.dataset_dir = Path(dataset_dir)
        self.n_splits = n_splits
        self.seed = seed
        self.test_ratio = test_ratio
        self.interaction_path = self.dataset_dir / "lastfm.inter"
        self.manifest_path = self.dataset_dir / self.MANIFEST_FILENAME

    def prepare(self):
        """
        Generate the split files, or reuse them when the manifest matches.
        """
        source_hash = self._sha256(self.interaction_path)
        expected_files = self._expected_files()

        manifest = self._read_manifest()
        if self._can_reuse(manifest, source_hash, expected_files):
            return {**manifest["statistics"], "reused": True}

        header, interactions_by_user = self._read_interactions()
        development_rows, test_rows, validation_fold_by_row = self._split_by_user(interactions_by_user)

        self._write_atomic_file(self.dataset_dir / "lastfm.development.inter", header, development_rows)
        self._write_atomic_file(self.dataset_dir / "lastfm.test.inter", header, test_rows)
        self._write_atomic_file(self.dataset_dir / "lastfm.empty.inter", header, [])

        for fold in range(self.n_splits):
            train_rows = [
                row
                for row, validation_fold in validation_fold_by_row
                if validation_fold != fold
            ]
            valid_rows = [
                row
                for row, validation_fold in validation_fold_by_row
                if validation_fold == fold
            ]
            
            self._write_atomic_file(
                self.dataset_dir / f"lastfm.fold{fold}_train.inter",
                header,
                train_rows,
            )
            self._write_atomic_file(
                self.dataset_dir / f"lastfm.fold{fold}_valid.inter",
                header,
                valid_rows,
            )

        statistics = {
            "users": len(interactions_by_user),
            "development_interactions": len(development_rows),
            "test_interactions": len(test_rows),
            "folds": self.n_splits,
        }
        
        self._write_manifest(source_hash, expected_files, statistics)
        
        return {**statistics, "reused": False}

    def fold_benchmark(self, fold):
        return [f"fold{fold}_train", f"fold{fold}_valid", "empty"]

    @staticmethod
    def final_benchmark() -> list[str]:
        return ["development", "empty", "test"]

    def _split_by_user(self, interactions_by_user):
        """
        Split the interactions into development and test sets, and assign each development interaction to a fold.

        Args:
            interactions_by_user: A dictionary mapping user IDs to their interactions.

        Returns:
            header: The header row of the interaction file.
            development_rows: The development interactions.
            test_rows: The test interactions.
            validation_fold_by_row: A list of tuples containing the interaction and its validation fold.
        """
        development_rows = []
        test_rows = []
        validation_fold_by_row = []
        
        random_generator = random.Random(self.seed)

        for user_id in sorted(interactions_by_user):
            user_rows = list(interactions_by_user[user_id])
            random_generator.shuffle(user_rows)
            
            test_count = max(1, math.floor(len(user_rows) * self.test_ratio))
            
            user_test_rows = user_rows[:test_count]
            user_development_rows = user_rows[test_count:]
            
            if len(user_development_rows) < self.n_splits:
                raise ValueError(
                    f"User {user_id} has only {len(user_development_rows)} development "
                    f"interactions for {self.n_splits} folds"
                )

            test_rows.extend(user_test_rows)
            development_rows.extend(user_development_rows)
            
            for index, row in enumerate(user_development_rows):
                validation_fold_by_row.append((row, index % self.n_splits))

        return development_rows, test_rows, validation_fold_by_row

    def _read_interactions(self):
        """
        Read the interactions from the atomic file and group them by user.

        Returns:
            header: The header row of the interaction file.
            interactions_by_user: A dictionary mapping user IDs to their interactions.
        """
        interactions_by_user = defaultdict(list)
        
        with self.interaction_path.open(encoding="utf-8", newline="") as input_file:
            reader = csv.reader(input_file, delimiter="\t")
            header = next(reader, None)
            
            for row in reader:
                interactions_by_user[row[0]].append(row)
        
        return header, dict(interactions_by_user)

    def _expected_files(self):
        files = [
            self.dataset_dir / "lastfm.development.inter",
            self.dataset_dir / "lastfm.test.inter",
            self.dataset_dir / "lastfm.empty.inter",
        ]
        
        for fold in range(self.n_splits):
            files.extend([
                self.dataset_dir / f"lastfm.fold{fold}_train.inter",
                self.dataset_dir / f"lastfm.fold{fold}_valid.inter",
            ])
        
        return files

    def _can_reuse(self, manifest, source_hash, expected_files):
        """
        Reuses an already processed folder split.

        Args:
            manifest: The manifest file.
            source_hash: The hash of the source file.
            expected_files: The list of expected files.

        Returns:
            bool: True if the split can be reused, False otherwise.
        """
        if manifest is None:
            return False
        
        return (
            manifest.get("source_sha256") == source_hash
            and manifest.get("n_splits") == self.n_splits
            and manifest.get("seed") == self.seed
            and manifest.get("test_ratio") == self.test_ratio
            and all(path.is_file() for path in expected_files)
            and "statistics" in manifest
        )

    def _read_manifest(self):
        """
        Reads the manifest file.

        Returns:
            dict: The manifest data, or None if the file doesn't exist or is invalid.
        """
        if not self.manifest_path.is_file():
            return None
        
        try:
            with self.manifest_path.open(encoding="utf-8") as input_file:
                return json.load(input_file)
        except (json.JSONDecodeError, OSError):
            return None

    def _write_manifest(self, source_hash, expected_files, statistics):
        manifest = {
            "source_sha256": source_hash,
            "n_splits": self.n_splits,
            "seed": self.seed,
            "test_ratio": self.test_ratio,
            "files": [path.name for path in expected_files],
            "statistics": statistics,
        }
        
        with self.manifest_path.open("w", encoding="utf-8") as output_file:
            json.dump(manifest, output_file, indent=2)
            output_file.write("\n")

    @staticmethod
    def _write_atomic_file(path, header, rows):
        with path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.writer(output_file, delimiter="\t", lineterminator="\n")
            
            writer.writerow(header)
            writer.writerows(rows)

    @staticmethod
    def _sha256(path):
        digest = hashlib.sha256()
        
        with path.open("rb") as input_file:
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                digest.update(chunk)
        
        return digest.hexdigest()
