import argparse
from pathlib import Path

from src.evaluators.multivae_evaluator import MultiVAEEvaluator
from src.splitters.lastfm_cross_val import LastFMCrossValidationSplitter
from src.splitters.yelp_cross_val import YelpCrossValidationSplitter
from src.utils.console import ConsoleColor, styled_print

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

# ----- Config
DATASETS = {
    "lastfm": {
        "config": "multivae.yaml",
        "splitter": LastFMCrossValidationSplitter,
    },
    "yelp": {
        "config": "multivae_yelp.yaml",
        "splitter": YelpCrossValidationSplitter,
    },
}
# -----


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train and evaluate MultiVAE on sampled datasets."
    )

    parser.add_argument(
        "--dataset",
        choices=("all", *DATASETS),
        default="all",
        help="Dataset to evaluate (default: all).",
    )
    
    parser.add_argument(
        "--user-limit",
        type=int,
        default=1000,
        help="Number of sampled users (default: 1000).",
    )

    parser.add_argument(
        "--item-limit",
        type=int,
        default=1000,
        help="Number of sampled items (default: 1000).",
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

    hyperparameter_config_path = (
        REPOSITORY_ROOT / "config/hyperparameters/multivae.yaml"
    )
    dataset_names = DATASETS if arguments.dataset == "all" else (arguments.dataset,)

    for dataset_index, dataset_name in enumerate(dataset_names):
        settings = DATASETS[dataset_name]

        styled_print(
            f"===== EVALUATING MultiVAE ON {dataset_name.upper()} =====",
            ConsoleColor.YELLOW,
            bold=True,
        )

        dataset_dir = REPOSITORY_ROOT / "data/sample" / dataset_name
        config_path = REPOSITORY_ROOT / "config/models" / settings["config"]

        evaluator = MultiVAEEvaluator(
            dataset_dir=dataset_dir,
            user_limit=arguments.user_limit,
            item_limit=arguments.item_limit,
            config_path=config_path,
            hp_search_config_path=hyperparameter_config_path,
            cross_validation_splitter=settings["splitter"],
        )

        evaluator.evaluate(
            cross_validation=arguments.cross_validation,
            hyperparameter_search=arguments.hyperparameter_search,
            n_splits=arguments.folds,
            estimate_runtime=dataset_index == 0,
            dataset_count=len(dataset_names),
        )


if __name__ == "__main__":
    main()
