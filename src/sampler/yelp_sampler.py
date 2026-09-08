from pathlib import Path

from src.sampler.dataset_sampler_interface import IDatasetSampler

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class YelpSampler(IDatasetSampler):
    """Create a reproducible Yelp sample in RecBole's atomic format."""

    DATASET_NAME = "yelp"
