import argparse
from pathlib import Path

from src.data.yelp import YelpTransformDataset
from src.utils.console import styled_print

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Transform the Yelp Open Dataset into RecBole atomic files."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data/raw/yelp",
        help="Directory containing the raw Yelp JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data/processed/yelp",
        help="Directory where the RecBole atomic files will be written.",
    )
    parser.add_argument(
        "--use-restaurants-users-only",
        action="store_true",
        help="Keep only users whose predominant preference is restaurants or food.",
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    
    styled_print(
        "Transforming the Yelp Open Dataset into RecBole atomic files...",
        bold=True,
    )
    
    transformer = YelpTransformDataset(
        arguments.raw_dir,
        arguments.output_dir,
        use_restaurants_users_only=arguments.use_restaurants_users_only,
    )
    statistics = transformer.transform()

    styled_print(
        f"\nYelp atomic files created in {arguments.output_dir.resolve()}",
        bold=True,
    )
    for name, value in statistics.items():
        styled_print(f"{name}: {value}")
    print()


if __name__ == "__main__":
    main()
