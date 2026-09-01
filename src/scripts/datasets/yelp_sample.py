import argparse
from pathlib import Path

from src.sampler.yelp_sampler import YelpSampler
from src.utils.console import styled_print

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Create a reproducible Yelp sample."
    )

    parser.add_argument(
        "--source-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data/processed/yelp",
        help="Directory containing the transformed Yelp atomic files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data/sample/yelp",
        help="Directory where the sampled atomic files will be written.",
    )
    parser.add_argument("--user-limit", type=int, default=1000)
    parser.add_argument("--item-limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-user-interactions", type=int, default=6)

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    styled_print("Sampling the Yelp dataset...", bold=True)

    sampler = YelpSampler(
        source_dir=arguments.source_dir,
        output_dir=arguments.output_dir,
        user_limit=arguments.user_limit,
        item_limit=arguments.item_limit,
        seed=arguments.seed,
        minimum_user_interactions=arguments.minimum_user_interactions,
    )
    statistics = sampler.create_sample()

    styled_print(
        f"\nYelp sample created in {arguments.output_dir.resolve()}",
        bold=True,
    )
    for name, value in statistics.items():
        styled_print(f"{name}: {value}")
    print()


if __name__ == "__main__":
    main()
