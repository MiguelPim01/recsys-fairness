from pathlib import Path

from src.sampler.dataset_sampler_interface import IDatasetSampler

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class YelpSampler(IDatasetSampler):
    """Create a reproducible Yelp sample in RecBole's atomic format."""

    DATASET_NAME = "yelp"

    def __init__(
        self,
        source_dir=REPOSITORY_ROOT / "data/processed/yelp",
        output_dir=REPOSITORY_ROOT / "data/sample/yelp",
        user_limit=50,
        item_limit=50,
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
