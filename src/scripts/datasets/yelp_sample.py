import argparse
from pathlib import Path

from src.sampler.yelp_sampler import YelpSampler
from src.utils.sample_statistics import generate_sample_statistics

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

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPOSITORY_ROOT / "results",
        help="Root directory where sample statistics will be written.",
    )
    
    parser.add_argument(
        "--user-limit", 
        type=int, 
        default=1000
    )
    
    parser.add_argument(
        "--item-limit", 
        type=int, 
        default=1000
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42
    )
    
    parser.add_argument(
        "--minimum-user-interactions", 
        type=int, 
        default=6
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()
    
    print("=" * 70)
    print("= 4. Sampling the Yelp dataset")
    print("=" * 70 + "\n")

    sampler = YelpSampler(
        source_dir=arguments.source_dir,
        output_dir=arguments.output_dir,
        user_limit=arguments.user_limit,
        item_limit=arguments.item_limit,
        seed=arguments.seed,
        minimum_user_interactions=arguments.minimum_user_interactions,
    )
    statistics = sampler.create_sample()

    statistics_output_dir = (
        arguments.results_dir
        / f"{arguments.user_limit}_{arguments.item_limit}"
        / "yelp"
        / "sample_statistics"
    )
    generate_sample_statistics(
        sample_dir=arguments.output_dir,
        dataset="yelp",
        output_dir=statistics_output_dir,
    )

    print(f"\nYelp sample created in {arguments.output_dir.resolve()}")
    print(f"Sample statistics created in {statistics_output_dir.resolve()}")
    
    for name, value in statistics.items():
        print(f"\t - {name}: {value}")
    print()


if __name__ == "__main__":
    main()
