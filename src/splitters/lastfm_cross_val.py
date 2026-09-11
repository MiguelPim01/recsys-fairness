import math

from src.splitters.cross_val_interface import ICrossValidationSplitter

# ----- Config
DATASET_NAME = "lastfm"
MANIFEST_FILENAME = "lastfm.cv_manifest.json"
# -----

class LastFMCrossValidationSplitter(ICrossValidationSplitter):
    """Create reusable per-user folds and an isolated final test set."""

    DATASET_NAME = DATASET_NAME
    MANIFEST_FILENAME = MANIFEST_FILENAME
    MANIFEST_VERSION = 2
    REQUIRES_EXTERNAL_SPLIT = True

    @staticmethod
    def _preprocess_splits(header, development_rows, test_rows):
        try:
            play_count_index = header.index("play_count:float")
        except (AttributeError, ValueError) as error:
            if header is not None and "rating:float" in header:
                raise ValueError(
                    "The LastFM sample is already normalized. Recreate it before "
                    "preparing the evaluation splits."
                ) from error

            raise ValueError(
                "The LastFM interaction file must contain play_count:float"
            ) from error

        logged_counts = []

        for row in development_rows:
            logged_counts.append(
                LastFMCrossValidationSplitter._logged_play_count(
                    row[play_count_index]
                )
            )

        if not logged_counts:
            raise ValueError("The LastFM development set has no interactions")

        minimum = min(logged_counts)
        maximum = max(logged_counts)

        for row in development_rows + test_rows:
            logged_count = LastFMCrossValidationSplitter._logged_play_count(
                row[play_count_index]
            )

            if math.isclose(minimum, maximum):
                rating = 3.0
            else:
                rating = 1.0 + 4.0 * (logged_count - minimum) / (maximum - minimum)
                rating = min(5.0, max(1.0, rating))

            row[play_count_index] = rating

        normalized_header = list(header)
        normalized_header[play_count_index] = "rating:float"

        return normalized_header, {
            "method": "log1p_min_max",
            "source_field": "play_count",
            "target_field": "rating",
            "development_log_min": minimum,
            "development_log_max": maximum,
            "target_min": 1.0,
            "target_max": 5.0,
            "clip": True,
        }

    @staticmethod
    def _logged_play_count(value):
        try:
            play_count = float(value)
        except ValueError as error:
            raise ValueError(f"Invalid LastFM play count: {value}") from error

        if not math.isfinite(play_count) or play_count <= 0:
            raise ValueError(
                f"LastFM play count must be finite and positive: {play_count}"
            )

        return math.log1p(play_count)
