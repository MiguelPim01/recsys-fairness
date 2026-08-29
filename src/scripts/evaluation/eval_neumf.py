import argparse
from pathlib import Path

from src.evaluators.neumf_evaluator import NeuMFEvaluator
from src.sampler.lastfm_sampler import LastFMSampler
from src.splitters.lastfm_cross_val import LastFMCrossValidationSplitter

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def parse_arguments():
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

def main():
    arguments = parse_arguments()
    
    # TODO: Temporário. Após o filtro estar correto irei tirar essa parte.
    sampler = LastFMSampler()
    statistics = sampler.create_sample()
    
    print(
        f"LastFM sample: {statistics['users']} users | "
        f"{statistics['items']} items | "
        f"{statistics['interactions']} interactions"
    )
    
    dataset_dir = REPOSITORY_ROOT / "data/sample/lastfm"
    config_path = REPOSITORY_ROOT / "config/models/neumf.yaml"
    hyperparameter_config_path = REPOSITORY_ROOT / "config/hyperparameters/neumf.yaml"

    # Creating and running evaluation
    evaluator = NeuMFEvaluator(
        dataset_dir=dataset_dir,
        config_path=config_path,
        hp_search_config_path=hyperparameter_config_path,
        cross_validation_splitter=LastFMCrossValidationSplitter,
    )
    
    evaluator.evaluate(
        cross_validation=arguments.cross_validation,
        hyperparameter_search=arguments.hyperparameter_search,
        n_splits=arguments.folds,
    )


if __name__ == "__main__":
    main()
