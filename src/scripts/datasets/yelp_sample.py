import argparse
from pathlib import Path

from src.sampler.yelp_sampler import YelpSampler
from src.utils.experiments import experiment_seed, resolve_experiment, sample_root
from src.utils.sample_statistics import generate_sample_statistics
from src.utils.transforms import read_transform_variant

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
        "--experiment",
        required=True,
        help="Seed-based experiment identifier, for example seed_42.",
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
        "--minimum-user-interactions", 
        type=int, 
        default=6
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()
    experiment_dir = resolve_experiment(
        arguments.user_limit,
        arguments.item_limit,
        arguments.experiment,
    )
    seed = experiment_seed(experiment_dir)
    
    output_dir = sample_root(
        arguments.user_limit,
        arguments.item_limit,
        arguments.experiment,
    ) / "yelp"
    source_variant = read_transform_variant(arguments.source_dir, "yelp")
    
    print("=" * 70)
    print("= 4. Sampling the Yelp dataset")
    print("=" * 70 + "\n")

    sampler = YelpSampler(
        source_dir=arguments.source_dir,
        output_dir=output_dir,
        user_limit=arguments.user_limit,
        item_limit=arguments.item_limit,
        seed=seed,
        minimum_user_interactions=arguments.minimum_user_interactions,
        source_variant=source_variant,
    )
    statistics = sampler.create_sample()

    statistics_output_dir = experiment_dir / "yelp" / "sample_statistics"
    generate_sample_statistics(
        sample_dir=output_dir,
        dataset="yelp",
        output_dir=statistics_output_dir,
    )

    print(f"\nYelp sample available in {output_dir.resolve()}")
    print(f"Sample statistics created in {statistics_output_dir.resolve()}")
    
    for name, value in statistics.items():
        print(f"\t - {name}: {value}")
    print()


if __name__ == "__main__":
    main()
