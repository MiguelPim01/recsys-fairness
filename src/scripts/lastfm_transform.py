import argparse
from pathlib import Path

from src.data.lastfm import LastFMTransformDataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transform LastFM-360K into RecBole atomic files."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data/raw/lastfm_360k",
        help="Directory containing the raw LastFM-360K TSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data/processed/lastfm",
        help="Directory where the RecBole atomic files will be written.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    transformer = LastFMTransformDataset(arguments.raw_dir, arguments.output_dir)
    statistics = transformer.transform()

    print(f"LastFM atomic files created in {arguments.output_dir.resolve()}")
    for name, value in statistics.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
