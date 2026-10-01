import argparse
from pathlib import Path

from src.data.yelp import YelpTransformDataset
from src.utils.transforms import (
    can_reuse_transformation,
    invalidate_transform_manifest,
    write_transform_manifest,
)

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
    variant = {
        "use_restaurants_users_only": arguments.use_restaurants_users_only,
    }

    if can_reuse_transformation(arguments.output_dir, "yelp", variant):
        print(f"Reusing transformed Yelp data in {arguments.output_dir.resolve()}")
        return
    
    print("=" * 70)
    print("= 2. Transforming Yelp Open Dataset into RecBole atomic files")
    print("=" * 70 + "\n")
    
    invalidate_transform_manifest(arguments.output_dir, "yelp")
    transformer = YelpTransformDataset(
        arguments.raw_dir,
        arguments.output_dir,
        use_restaurants_users_only=arguments.use_restaurants_users_only,
    )
    statistics = transformer.transform()
    write_transform_manifest(arguments.output_dir, "yelp", variant)

    print(f"\nYelp atomic files created in {arguments.output_dir.resolve()}")
    
    for name, value in statistics.items():
        print(f"\t - {name}: {value}")
    print()


if __name__ == "__main__":
    main()
