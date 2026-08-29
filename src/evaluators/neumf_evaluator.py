import logging
import sys
import warnings
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import yaml
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, get_trainer, init_seed
from tqdm.auto import tqdm

from src.sampler.lastfm_cross_validation_splitter import LastFMCrossValidationSplitter

# ----- Config
LOGGER = logging.getLogger("recsys_fairness.evaluation")

HYPERPARAMETER_LABELS = {
    "learning_rate": "lr",
    "dropout_prob": "dropout",
    "mf_embedding_size": "mf_emb",
    "mlp_embedding_size": "mlp_emb",
}
# -----

class NeuMFEvaluator:
    """Train, cross-validate and tune RecBole's NeuMF."""

    def __init__(self, dataset_dir, config_path, hp_search_config_path):
        self.dataset_dir = Path(dataset_dir)
        self.config_path = Path(config_path)
        self.hp_search_config_path = Path(hp_search_config_path)

    def evaluate(self, cross_validation = False, hyperparameter_search = False, n_splits = 5):
        """
        Evaluate the NeuMF model.

        Args:
            cross_validation (bool, optional): _description_. Defaults to False.
            hyperparameter_search (bool, optional): _description_. Defaults to False.
            n_splits (int, optional): _description_. Defaults to 5.

        Returns:
            _type_: _description_
        """
        self._configure_project_logging()

        if not cross_validation and not hyperparameter_search:
            return self._evaluate_simple()

        base_config = self._build_config()
        
        splitter = LastFMCrossValidationSplitter(
            dataset_dir=self.dataset_dir,
            n_splits=n_splits,
            seed=base_config["seed"],
        )
        split_statistics = splitter.prepare()

        candidates = self._load_hyperparameter_candidates() if hyperparameter_search else [{}]
        
        fold_indexes = range(n_splits) if cross_validation else range(1)
        
        candidate_results = []
        validation_runs = len(candidates) * len(fold_indexes)
        validation_metric = base_config["valid_metric"]

        LOGGER.info(
            "Selection: %d users | %d development interactions | "
            "%d test interactions | %d runs | metric=%s\n",
            split_statistics["users"],
            split_statistics["development_interactions"],
            split_statistics["test_interactions"],
            validation_runs,
            validation_metric,
        )
        
        for candidate_index, hyperparameters in enumerate(candidates, start=1):
            LOGGER.info(
                "Candidate %d/%d | %s",
                candidate_index,
                len(candidates),
                self._format_hyperparameters(hyperparameters),
            )
            
            fold_results = []
            
            progress = tqdm(fold_indexes, desc="  folds", unit="fold", dynamic_ncols=True,)
            for fold in progress:
                run = self._train_with_validation(
                    benchmark_filename=splitter.fold_benchmark(fold),
                    hyperparameters=hyperparameters,
                    run_seed=base_config["seed"] + fold,
                )
                
                fold_results.append({
                    "fold": fold,
                    "score": run["score"],
                    "metrics": run["metrics"],
                })
                
                progress.set_postfix(score=f"{run['score']:.4f}")

            aggregate = self._aggregate_fold_results(fold_results)
            
            candidate_result = {
                "hyperparameters": hyperparameters,
                "fold_results": fold_results,
                **aggregate,
            }
            candidate_results.append(candidate_result)
            
            LOGGER.info(
                "  %s: %.4f ± %.4f\n",
                validation_metric,
                aggregate["mean_score"],
                aggregate["std_score"],
            )

        best_candidate = self._select_best_candidate(
            candidate_results,
            bigger=base_config["valid_metric_bigger"],
        )
        
        LOGGER.info(
            "Selected | %s | %s %.4f ± %.4f",
            self._format_hyperparameters(best_candidate["hyperparameters"]),
            validation_metric,
            best_candidate["mean_score"],
            best_candidate["std_score"],
        )

        test_result = self._train_development_and_evaluate_test(
            splitter.final_benchmark(),
            best_candidate["hyperparameters"],
        )
        
        results = {
            "mode": {
                "cross_validation": cross_validation,
                "hyperparameter_search": hyperparameter_search,
            },
            "best_params": best_candidate["hyperparameters"],
            "validation": best_candidate,
            "candidates": candidate_results,
            "test_result": test_result,
        }
        
        LOGGER.info("Test | %s", self._format_metrics(test_result))
        
        return results

    def _evaluate_simple(self):
        """
        Evaluates with neither cross validation nor hyperparameter search.

        Returns:
            results (dict): Evaluation results.
        """
        config = self._build_config()
        
        init_seed(config["seed"], config["reproducibility"])

        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)
        
        _, trainer = self._create_model_and_trainer(config, train_data)

        LOGGER.info(
            "NeuMF: %d users | %d items | %d interactions | %d epoch(s)",
            dataset.user_num - 1,
            dataset.item_num - 1,
            len(dataset.inter_feat),
            config["epochs"],
        )
        
        best_valid_score, best_valid_result = trainer.fit(
            train_data,
            valid_data,
            saved=False,
            show_progress=False,
            verbose=False,
        )
        
        test_result = trainer.evaluate(
            test_data,
            load_best_model=False,
            show_progress=False,
        )

        validation = {
            "hyperparameters": {},
            "fold_results": [{
                "fold": 0,
                "score": float(best_valid_score),
                "metrics": best_valid_result,
            }],
            "mean_score": float(best_valid_score),
            "std_score": 0.0,
            "mean_metrics": dict(best_valid_result),
            "std_metrics": {key: 0.0 for key in best_valid_result},
        }
        
        results = {
            "mode": {
                "cross_validation": False,
                "hyperparameter_search": False,
            },
            "best_params": {},
            "validation": validation,
            "candidates": [validation],
            "test_result": test_result,
        }
        
        LOGGER.info("Validation | %s", self._format_metrics(best_valid_result))
        LOGGER.info("Test | %s", self._format_metrics(test_result))
        
        return results

    def _train_with_validation(self, benchmark_filename, hyperparameters, run_seed):
        """
        Trains the model with validation and returns the best score and metrics.

        Args:
            benchmark_filename (list[str]): _description_
            hyperparameters (dict[str, Any]): _description_
            run_seed (int): _description_

        Returns:
            dict[str, Any]: _description_
        """
        config = self._build_config({
            "benchmark_filename": benchmark_filename,
            "seed": run_seed,
            "show_progress": False,
            **hyperparameters,
        })
        
        init_seed(config["seed"], config["reproducibility"])
        
        dataset = create_dataset(config)
        train_data, valid_data, _ = data_preparation(config, dataset)
        
        _, trainer = self._create_model_and_trainer(config, train_data)
        
        best_score, best_result = trainer.fit(
            train_data,
            valid_data,
            saved=False,
            show_progress=False,
            verbose=False,
        )
        
        return {"score": float(best_score), "metrics": best_result}

    def _train_development_and_evaluate_test(self, benchmark_filename, hyperparameters):
        """
        Trains the model on the development data and evaluates it on the test data.

        Args:
            benchmark_filename (list[str]): _description_
            hyperparameters (dict[str, Any]): _description_

        Returns:
            _type_: _description_
        """
        config = self._build_config({
            "benchmark_filename": benchmark_filename,
            **hyperparameters,
        })
        
        init_seed(config["seed"], config["reproducibility"])
        
        dataset = create_dataset(config)
        train_data, _, test_data = data_preparation(config, dataset)
        
        _, trainer = self._create_model_and_trainer(config, train_data)

        LOGGER.info("Final training on all development interactions")
        
        trainer.fit(
            train_data,
            valid_data=None,
            saved=False,
            show_progress=False,
            verbose=False,
        )
        
        return trainer.evaluate(
            test_data,
            load_best_model=False,
            show_progress=False,
        )

    def _build_config(self, overrides = None) -> Config:
        """
        Builds a configuration object for RecBole's NeuMF model.

        Args:
            overrides (dict, optional): Overide configuration. Defaults to None.

        Returns:
            Config: Configuration object for RecBole's NeuMF model.
        """
        
        config_dict = {
            "data_path": str(self.dataset_dir.parent.resolve()),
            **(overrides or {}),
        }
        
        # patch will make Config RecBole consider that there is no argv
        with patch.object(sys, "argv", [sys.argv[0]]):
            return Config(
                model="NeuMF",
                dataset=self.dataset_dir.name,
                config_file_list=[str(self.config_path)],
                config_dict=config_dict,
            )

    @staticmethod
    def _create_model_and_trainer(config: Config, train_data):
        """
        Creates a model and trainer for the NeuMF model.

        Args:
            config (Config): RecBole configuration.
            train_data: Training data prepared by RecBole.

        Returns:
            model: The NeuMF model.
            trainer: The trainer for the NeuMF model. 
        """
        init_seed(config["seed"], config["reproducibility"])
        
        model = get_model(config["model"])(config, train_data.dataset).to(
            config["device"]
        )
        
        trainer_class = get_trainer(config["MODEL_TYPE"], config["model"])
        trainer = trainer_class(config, model)
        
        return model, trainer

    def _load_hyperparameter_candidates(self) -> list[dict[str, Any]]:
        """
        Loads the hyperparameter candidates from the search configuration file.

        Returns:
            candidates (list[dict[str, Any]]): A list of hyperparameter configurations.
        """ 
        with self.hp_search_config_path.open(encoding="utf-8") as input_file:
            hp_search_config = yaml.safe_load(input_file)
        
        candidates = hp_search_config.get("configurations") if hp_search_config else None
        
        return candidates

    @staticmethod
    def _aggregate_fold_results(fold_results):
        """
        Aggregates the results from multiple folds into a single result.

        Args:
            fold_results (list[dict[str, Any]]): List of results from each fold

        Returns:
            dict[str, Any]: List of results from each fold
        """
        scores = np.asarray([result["score"] for result in fold_results], dtype=float)
        metric_names = list(fold_results[0]["metrics"])
        
        mean_metrics = {}
        std_metrics = {}
        
        for metric_name in metric_names:
            values = np.asarray(
                [result["metrics"][metric_name] for result in fold_results],
                dtype=float,
            )
            
            mean_metrics[metric_name] = float(np.mean(values))
            std_metrics[metric_name] = float(np.std(values))
        
        return {
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "mean_metrics": mean_metrics,
            "std_metrics": std_metrics,
        }

    @staticmethod
    def _select_best_candidate(candidates, bigger):
        """
        Selects best hyperparameter set.

        Args:
            candidates (list[dict[str, Any]]): Hyperparameter's options.
            bigger (bool): If True, selects higher mean_score, otherwise selects lower mean_score.

        Returns:
            best (dict[str, Any]): Hyperparameter set with the best mean_score.
        """
        best = candidates[0]
        
        for candidate in candidates[1:]:
            is_better = (
                candidate["mean_score"] > best["mean_score"]
                if bigger
                else candidate["mean_score"] < best["mean_score"]
            )
            
            if is_better:
                best = candidate
        
        return best

    @staticmethod
    def _format_hyperparameters(hyperparameters):
        if not hyperparameters:
            return "NeuMF.yaml defaults"
        
        return ", ".join(
            f"{HYPERPARAMETER_LABELS.get(name, name)}={value}"
            for name, value in hyperparameters.items()
        )

    @staticmethod
    def _format_metrics(metrics):
        return " | ".join(f"{name}={float(value):.4f}" for name, value in metrics.items())

    @staticmethod
    def _configure_project_logging():
        # Forces global log config for RecBole
        logging.basicConfig(level=logging.WARNING, force=True)
        
        # Clears handlers and create new one
        LOGGER.handlers.clear()
        
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        
        LOGGER.addHandler(handler)
        
        # Sets LOGGER level to INFO
        LOGGER.setLevel(logging.INFO)
        LOGGER.propagate = False
        
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"recbole\..*")
