import argparse

from src.evaluators.neumf_evaluator import NeuMFEvaluator
from src.sampler.lastfm_sampler import LastFMSampler


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate NeuMF on the LastFM sample."
    )
    parser.add_argument(
        "--cross-validation",
        action="store_true",
        help="Evaluate the configuration across user-stratified folds.",
    )
    parser.add_argument(
        "--hyperparameter-search",
        action="store_true",
        help="Evaluate the configurations from the hyperparameter YAML.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5).",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    sampler = LastFMSampler()
    statistics = sampler.create_sample()
    print(
        f"LastFM sample: {statistics['users']} users | "
        f"{statistics['items']} items | "
        f"{statistics['interactions']} interactions"
    )

    evaluator = NeuMFEvaluator()
    evaluator.evaluate(
        cross_validation=arguments.cross_validation,
        hyperparameter_search=arguments.hyperparameter_search,
        n_splits=arguments.folds,
    )


if __name__ == "__main__":
    main()
