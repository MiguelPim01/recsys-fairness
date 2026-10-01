"""Rebuild fairness artifacts from immutable experiment checkpoints."""

import argparse
import warnings
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_trainer, init_seed

from src.fairness import GroupFairnessAnalyzer
from src.models.multivae import MultiVAE
from src.models.neumf import NeuMF
from src.utils.experiments import (
    resolve_experiment,
    validate_experiment_dataset,
    validate_model_checkpoint,
)

MODELS = {
    "neumf": ("NeuMF", NeuMF),
    "multivae": ("MultiVAE", MultiVAE),
}
DATASETS = ("lastfm", "yelp")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument(
        "--dataset",
        choices=("all", *DATASETS),
        default="all",
    )
    parser.add_argument(
        "--model",
        choices=("all", *MODELS),
        default="all",
    )
    
    parser.add_argument("--user-limit", type=int, required=True)
    parser.add_argument("--item-limit", type=int, required=True)
    
    parser.add_argument(
        "--experiment",
        required=True,
        help="Seed-based experiment identifier, for example seed_42.",
    )
    
    return parser.parse_args()


def analyze_checkpoint(experiment_dir: Path, dataset_dir: Path, dataset_name: str, model_key: str):
    model_name, model_class = MODELS[model_key]
    dataset_output_dir = experiment_dir / dataset_name
    
    checkpoint_path = validate_model_checkpoint(
        experiment_dir,
        dataset_name,
        model_key,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint["config"]
    if str(config["dataset"]).casefold() != dataset_name:
        raise ValueError(
            f"Checkpoint dataset mismatch in {checkpoint_path}: "
            f"{config['dataset']} != {dataset_name}"
        )
    if str(config["model"]).casefold() != model_key:
        raise ValueError(
            f"Checkpoint model mismatch in {checkpoint_path}: "
            f"{config['model']} != {model_name}"
        )

    # RecBole normalizes ``data_path`` to the dataset directory while building
    # Config. A deserialized Config must therefore receive the final path here.
    config["data_path"] = str(dataset_dir.resolve())
    config["checkpoint_dir"] = str((dataset_output_dir / "models").resolve())
    config["fairness"]["output_dir"] = str(dataset_output_dir.resolve())

    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config["seed"], config["reproducibility"])
    model = model_class(config, train_data.dataset).to(config["device"])
    trainer_class = get_trainer(config["MODEL_TYPE"], config["model"])
    trainer = trainer_class(config, model)

    # RecBole 1.2.1 still references the NumPy alias removed in NumPy 1.24.
    with (
        patch.object(torch, "load", partial(torch.load, weights_only=False)),
        patch.object(np, "float", float, create=True),
    ):
        test_result = trainer.evaluate(
            test_data,
            load_best_model=True,
            model_file=str(checkpoint_path),
            show_progress=False,
        )

    analyzer = GroupFairnessAnalyzer(
        dataset_dir=dataset_dir,
        algorithm=model_name,
        config=config,
    )
    _, results_path = analyzer.analyze(
        trainer=trainer,
        test_data=test_data,
        development_data=(train_data, valid_data),
        recbole_metrics=test_result,
    )
    return results_path


def main():
    arguments = parse_arguments()
    
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"recbole\..*")
    
    experiment_dir = resolve_experiment(
        arguments.user_limit,
        arguments.item_limit,
        arguments.experiment,
    )
    
    datasets = DATASETS if arguments.dataset == "all" else (arguments.dataset,)
    models = MODELS if arguments.model == "all" else (arguments.model,)

    for dataset_name in datasets:
        dataset_dir = validate_experiment_dataset(experiment_dir, dataset_name)
        for model_key in models:
            print(
                f"Analyzing {model_key} on {dataset_name} "
                f"from {experiment_dir.name}"
            )
            results_path = analyze_checkpoint(
                experiment_dir,
                dataset_dir,
                dataset_name,
                model_key,
            )
            print(f"  results: {results_path}")


if __name__ == "__main__":
    main()
