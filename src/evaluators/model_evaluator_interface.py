import logging
import multiprocessing
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import patch

import numpy as np
import torch
import yaml
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, get_trainer, init_seed
from tqdm.auto import tqdm

from src.fairness import GroupFairnessAnalyzer

# ----- Config
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger("recsys_fairness.evaluation")
# -----

class IModelEvaluator:
    """Train, cross-validate and tune a RecBole model."""

    MODEL_NAME = None
    MODEL_CLASS = None
    HYPERPARAMETER_LABELS: ClassVar[dict] = {}

    def __init__(self, dataset_dir, user_limit, item_limit, config_path, hp_search_config_path, cross_validation_splitter = None):
        self.dataset_dir = Path(dataset_dir)
        self.config_path = Path(config_path)
        self.hp_search_config_path = Path(hp_search_config_path)
        self.cross_validation_splitter = cross_validation_splitter
        self.user_limit = user_limit
        self.item_limit = item_limit

    def evaluate(self, cross_validation=False, hyperparameter_search=False, n_splits=5, estimate_runtime=False, dataset_count=1, fold_workers=1):
        """
        Evaluate the model.

        Args:
            cross_validation (bool, optional): Whether to perform cross-validation. Defaults to False.
            hyperparameter_search (bool, optional): Whether to perform hyperparameter search. Defaults to False.
            n_splits (int, optional): The number of splits for cross-validation. Defaults to 5.
            estimate_runtime (bool, optional): Whether to estimate the model's total training time from its first fold. Defaults to False.
            dataset_count (int, optional): Number of datasets included in the model execution. Defaults to 1.
            fold_workers (int, optional): Maximum number of folds to run in parallel. Defaults to 1.

        Returns:
            results (dict): Evaluation results, including best hyperparameters, validation results, and test results. 
        """
        self._configure_project_logging()

        if fold_workers < 1:
            raise ValueError("fold_workers must be greater than or equal to 1")

        base_config = self._build_config()

        # 1. If there aren't any cv and hyperparameter search, run simple evaluation.
        if (
            not cross_validation
            and not hyperparameter_search
            and not self.cross_validation_splitter.REQUIRES_EXTERNAL_SPLIT
        ):
            return self._evaluate_simple()
        
        splitter = self.cross_validation_splitter(
            dataset_dir=self.dataset_dir,
            n_splits=n_splits,
            seed=base_config["seed"],
        )
        split_statistics = splitter.prepare()

        candidates = self._load_hyperparameter_candidates() if hyperparameter_search else [{}]
        
        fold_indexes = list(range(n_splits)) if cross_validation else [0]
        effective_fold_workers = min(fold_workers, len(fold_indexes))
        total_training_runs = dataset_count * (
            len(candidates) * len(fold_indexes) + 1
        )
        
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

        LOGGER.info("Fold workers: %d\n", effective_fold_workers)
        
        # 2. If there are candidates for hyperparameter search, runs the loop
        executor = None

        if effective_fold_workers > 1:
            executor = ProcessPoolExecutor(
                max_workers=effective_fold_workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=self._configure_project_logging,
            )

        try:
            for candidate_index, hyperparameters in enumerate(candidates, start=1):
                LOGGER.info(
                    "Candidate %d/%d | %s",
                    candidate_index,
                    len(candidates),
                    self._format_hyperparameters(hyperparameters),
                )

                fold_results, first_fold_elapsed = self._evaluate_folds(
                    splitter=splitter,
                    fold_indexes=fold_indexes,
                    hyperparameters=hyperparameters,
                    base_seed=base_config["seed"],
                    executor=executor,
                )

                if estimate_runtime and candidate_index == 1:
                    parallel_validation_runs = (
                        dataset_count * len(candidates) * len(fold_indexes)
                        / effective_fold_workers
                    )
                    estimated_runs = parallel_validation_runs + dataset_count
                    elapsed_hours = first_fold_elapsed * estimated_runs / 3600

                    LOGGER.info(
                        "\n --> Estimativa de duração do %s: %.2f horas "
                        "(%d treinos | %d workers)\n",
                        self.MODEL_NAME,
                        elapsed_hours,
                        total_training_runs,
                        effective_fold_workers,
                    )

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
        finally:
            if executor is not None:
                executor.shutdown(cancel_futures=True)

        best_candidate = self._select_best_candidate(
            candidate_results,
            bigger=base_config["valid_metric_bigger"],
        )
        
        LOGGER.info("Selected Hyperparameters:")
        LOGGER.info("  --> %s", self._format_hyperparameters(best_candidate["hyperparameters"]))
        LOGGER.info(
            "  --> %s = %.4f ± %.4f\n", 
            validation_metric,
            best_candidate["mean_score"],
            best_candidate["std_score"]
        )

        # 3. Final evaluation: trains model with development data and evaluates on test data.
        test_result, analysis, _ = self._train_development_and_evaluate_test(
            splitter.final_benchmark(),
            best_candidate["hyperparameters"],
            best_candidate["median_epoch"],
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
            "analysis": analysis,
        }
        
        LOGGER.info("Test Results: \n  --> %s\n", self._format_metrics(test_result))
        
        return results

    def _evaluate_folds(self, splitter, fold_indexes, hyperparameters, base_seed, executor):
        """
        Evaluates every fold sequentially or using the provided process pool.

        Args:
            splitter: The cross-validation splitter.
            fold_indexes (list[int]): Fold indexes to evaluate.
            hyperparameters (dict[str, Any]): Hyperparameters to override the default configuration.
            base_seed (int): Base seed used to initialize each fold.
            executor (ProcessPoolExecutor, optional): Process pool used to run folds in parallel.

        Returns:
            fold_results: Results ordered by fold index.
            first_fold_elapsed: Elapsed time for the first fold.
        """
        fold_results = []
        first_fold = fold_indexes[0]
        start_time = time.perf_counter()

        progress = tqdm(
            fold_indexes,
            desc="  folds",
            unit="fold",
            dynamic_ncols=True,
        )

        if executor is None:
            try:
                for fold in progress:
                    result = self._evaluate_fold(
                        fold=fold,
                        benchmark_filename=splitter.fold_benchmark(fold),
                        hyperparameters=hyperparameters,
                        run_seed=base_seed + fold,
                    )
                    fold_results.append(result)
                    progress.set_postfix(score=f"{result['score']:.4f}")

                    if fold == first_fold:
                        first_fold_elapsed = time.perf_counter() - start_time
            except BaseException:
                progress.close()
                raise
        else:
            futures = {
                executor.submit(
                    self._evaluate_fold,
                    fold,
                    splitter.fold_benchmark(fold),
                    hyperparameters,
                    base_seed + fold,
                ): fold
                for fold in fold_indexes
            }

            try:
                for future in as_completed(futures):
                    result = future.result()
                    fold_results.append(result)
                    progress.update()
                    progress.set_postfix(score=f"{result['score']:.4f}")

                    if result["fold"] == first_fold:
                        first_fold_elapsed = time.perf_counter() - start_time
            except BaseException:
                for future in futures:
                    future.cancel()
                progress.close()
                raise

        progress.close()
        fold_results.sort(key=lambda result: result["fold"])

        return fold_results, first_fold_elapsed

    def _evaluate_fold(self, fold, benchmark_filename, hyperparameters, run_seed):
        """
        Trains and evaluates one validation fold.

        Args:
            fold (int): Fold index.
            benchmark_filename (list[str]): File extensions that RecBole will use.
            hyperparameters (dict[str, Any]): Hyperparameters to override the default configuration.
            run_seed (int): Seed.

        Returns:
            result (dict[str, Any]): Fold index, best epoch, score and metrics.
        """
        run = self._train_with_validation(
            benchmark_filename=benchmark_filename,
            hyperparameters=hyperparameters,
            run_seed=run_seed,
        )

        return {
            "fold": fold,
            "epoch": run["epoch"],
            "score": run["score"],
            "metrics": run["metrics"],
        }

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

        best_epoch = 0

        def save_best_epoch(epoch, valid_score):
            nonlocal best_epoch

            if valid_score == trainer.best_valid_score:
                best_epoch = epoch + 1

        LOGGER.info(
            "%s: %d users | %d items | %d interactions | %d epoch(s)",
            self.MODEL_NAME,
            dataset.user_num - 1,
            dataset.item_num - 1,
            len(dataset.inter_feat),
            config["epochs"],
        )
        
        best_valid_score, best_valid_result = trainer.fit(
            train_data,
            valid_data,
            saved=True,
            show_progress=False,
            verbose=False,
            callback_fn=save_best_epoch,
        )
        
        test_result = self._evaluate_saved_model(trainer, test_data)
        analysis, results_path = self._analyze_final_test(
            trainer=trainer,
            config=config,
            test_data=test_data,
            development_data=(train_data, valid_data),
            test_result=test_result,
        )

        validation = {
            "hyperparameters": {},
            "fold_results": [{
                "fold": 0,
                "epoch": best_epoch,
                "score": float(best_valid_score),
                "metrics": best_valid_result,
            }],
            "median_epoch": best_epoch,
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
            "analysis": analysis,
        }
        
        LOGGER.info("Validation | %s", self._format_metrics(best_valid_result))
        LOGGER.info("Test | %s", self._format_metrics(test_result))
        
        if results_path is not None:
            LOGGER.info("Fairness results | %s", results_path)
        
        return results

    def _train_with_validation(self, benchmark_filename, hyperparameters, run_seed):
        """
        Trains the model with validation and returns the best score and metrics.

        Args:
            benchmark_filename (list[str]): File extensions that RecBole will use.
            hyperparameters (dict[str, Any]): Hyperparameters to override the default configuration.
            run_seed (int): Seed.

        Returns:
            results (dict[str, Any]): Best epoch, best score, and metrics.
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

        best_epoch = 0

        def save_best_epoch(epoch, valid_score):
            nonlocal best_epoch

            if valid_score == trainer.best_valid_score:
                best_epoch = epoch + 1
        
        best_score, best_result = trainer.fit(
            train_data,
            valid_data,
            saved=False,
            show_progress=False,
            verbose=False,
            callback_fn=save_best_epoch,
        )
        
        return {
            "epoch": best_epoch,
            "score": float(best_score),
            "metrics": best_result,
        }

    def _train_development_and_evaluate_test(self, benchmark_filename, hyperparameters, epochs):
        """
        Trains the model on the development data and evaluates it on the test data.

        Args:
            benchmark_filename (list[str]): File extensions that RecBole will use.
            hyperparameters (dict[str, Any]): Hyperparameters to override the default configuration.
            epochs (int): The number of epochs to train the model.

        Returns:
            test_result: The evaluation results on the test data.
            analysis: The group-fairness analysis results.
            results_path: The path to the directory containing the analysis results.
        """
        config = self._build_config({
            "benchmark_filename": benchmark_filename,
            "epochs": epochs,
            **hyperparameters,
        })
        
        init_seed(config["seed"], config["reproducibility"])
        
        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)
        
        _, trainer = self._create_model_and_trainer(config, train_data)

        LOGGER.info(
            "Starting final evaluation on test set | %d epoch(s)",
            epochs,
        )
        
        trainer.fit(
            train_data,
            valid_data=None,
            saved=True,
            show_progress=False,
            verbose=False,
        )
        
        test_result = self._evaluate_saved_model(trainer, test_data)
        analysis, results_path = self._analyze_final_test(
            trainer=trainer,
            config=config,
            test_data=test_data,
            development_data=(train_data, valid_data),
            test_result=test_result,
        )
        
        return test_result, analysis, results_path

    def _build_config(self, overrides = None) -> Config:
        """
        Builds a configuration object for the RecBole model.

        Args:
            overrides (dict, optional): Overide configuration. Defaults to None.

        Returns:
            config (Config): Configuration object for the RecBole model.
        """
        
        config_dict = {
            "data_path": str(self.dataset_dir.parent.resolve()),
            **(overrides or {}),
        }
        
        # patch will make Config RecBole consider that there is no argv
        with patch.object(sys, "argv", [sys.argv[0]]):
            config = Config(
                model=self.MODEL_NAME,
                dataset=self.dataset_dir.name,
                config_file_list=[str(self.config_path)],
                config_dict=config_dict,
            )
        
        config["fairness"]["output_dir"] = str(self.results_dir)
        
        return config

    @classmethod
    def _create_model_and_trainer(cls, config: Config, train_data):
        """
        Creates a model and trainer.

        Args:
            config (Config): RecBole configuration.
            train_data: Training data prepared by RecBole.

        Returns:
            model: The model.
            trainer: The trainer for the model. 
        """
        init_seed(config["seed"], config["reproducibility"])
        
        model_class = cls.MODEL_CLASS or get_model(config["model"])
        model = model_class(config, train_data.dataset).to(config["device"])
        
        trainer_class = get_trainer(config["MODEL_TYPE"], config["model"])
        trainer = trainer_class(config, model)
        
        return model, trainer

    @staticmethod
    def _evaluate_saved_model(trainer, test_data):
        """
        Evaluates the saved model on the test data.

        Args:
            trainer: RecBole trainer.
            test_data: Test data prepared by RecBole.

        Returns:
            dict[str, Any]: Test results.
        """
        with patch.object(torch, "load", partial(torch.load, weights_only=False)):
            return trainer.evaluate(
                test_data,
                load_best_model=True,
                show_progress=False,
            )

    def _analyze_final_test(self, trainer, config, test_data, development_data, test_result):
        """
        Run detailed group-fairness analysis only for the final test.
        
        Args:
            trainer: RecBole trainer.
            config: RecBole configuration.
            test_data: Test data prepared by RecBole.
            development_data: Tuple of (train_data, valid_data) prepared by RecBole.
            test_result: Test results.
        
        Returns:
            analysis: The group-fairness analysis results.
            results_path: The path to the directory containing the analysis results.
        """
        settings = config["fairness"]
        if not settings or not settings.get("enabled", False):
            return None, None

        analyzer = GroupFairnessAnalyzer(
            dataset_dir=self.dataset_dir,
            algorithm=self.MODEL_NAME,
            config=config,
        )
        
        analysis, results_path = analyzer.analyze(
            trainer=trainer,
            test_data=test_data,
            development_data=development_data,
            recbole_metrics=test_result,
        )
        
        return analysis, results_path

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
        epochs = np.asarray([result["epoch"] for result in fold_results], dtype=int)
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
            "median_epoch": int(np.median(epochs)),
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

    def _format_hyperparameters(self, hyperparameters):
        if not hyperparameters:
            return f"{self.MODEL_NAME}.yaml defaults"
        
        return ", ".join(
            f"{self.HYPERPARAMETER_LABELS.get(name, name)}={value}"
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
    
    @property
    def results_dir(self) -> Path:
        return REPOSITORY_ROOT / "results" / (
            f"{self.user_limit}_{self.item_limit}"
        )
