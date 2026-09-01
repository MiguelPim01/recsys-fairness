"""
Transform LastFM-360K into RecBole atomic files.
"""

import argparse
from pathlib import Path

from src.data.lastfm import LastFMTransformDataset
from src.utils.console import styled_print

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def parse_arguments():
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


def main():
    arguments = parse_arguments()
    
    styled_print(
        "Transforming LastFM-360K into RecBole atomic files...",
        bold=True,
    )
    
    transformer = LastFMTransformDataset(arguments.raw_dir, arguments.output_dir)
    statistics = transformer.transform()

    styled_print(
        f"\nLastFM atomic files created in {arguments.output_dir.resolve()}",
        bold=True,
    )
    for name, value in statistics.items():
        styled_print(f"{name}: {value}")
    print()


if __name__ == "__main__":
    main()
