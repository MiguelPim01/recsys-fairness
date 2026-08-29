import argparse
from pathlib import Path

from src.data.yelp import YelpTransformDataset

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
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    
    print("Transforming the Yelp Open Dataset into RecBole atomic files...")
    
    transformer = YelpTransformDataset(arguments.raw_dir, arguments.output_dir)
    statistics = transformer.transform()

    print(f"\nYelp atomic files created in {arguments.output_dir.resolve()}")
    for name, value in statistics.items():
        print(f"{name}: {value}")
    print()


if __name__ == "__main__":
    main()
