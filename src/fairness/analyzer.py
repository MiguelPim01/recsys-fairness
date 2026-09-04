import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from src.fairness.grouping import (
    Partition,
    UserProfile,
    latent_partitions,
    metadata_partitions,
)
from src.fairness.results import ResultsStore


@dataclass(frozen=True)
class UserEvaluation:
    user_id: str
    ndcg: dict[str, float]
    squared_errors: tuple[float, ...]


class GroupFairnessAnalyzer:
    """Calculate detailed ranking and group-fairness metrics on final test data."""

    def __init__(
        self,
        dataset_dir: str | Path,
        algorithm: str,
        config,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.algorithm = algorithm
        self.config = config
        self.settings = dict(config["fairness"])

    def analyze(
        self,
        trainer,
        test_data,
        development_data: Iterable,
        recbole_metrics: dict[str, Any],
    ) -> tuple[dict[str, Any], Path]:
        topk = sorted({int(value) for value in self.config["topk"]})
        if not topk or topk[0] <= 0:
            raise ValueError("topk must contain positive integers")

        activity_counts = self._development_activity(development_data)
        raw_profiles = self._read_profiles()
        evaluations = self._evaluate_users(trainer, test_data, topk)
        profiles = []
        for evaluation in evaluations:
            try:
                raw_profile = raw_profiles[evaluation.user_id]
            except KeyError as error:
                raise ValueError(
                    f"Missing profile for test user {evaluation.user_id}"
                ) from error

            profiles.append(
                UserProfile(
                    user_id=evaluation.user_id,
                    development_interactions=activity_counts[evaluation.user_id],
                    gender=raw_profile.get("gender", ""),
                    age=raw_profile.get("age"),
                    is_active=raw_profile.get("is_active"),
                    friend_count=raw_profile.get("friend_count"),
                    fans=raw_profile.get("fans"),
                    tenure_years=raw_profile.get("tenure_years"),
                )
            )

        if any(profile.development_interactions <= 0 for profile in profiles):
            raise ValueError("Every test user must have development interactions")

        activity_fraction = float(self.settings["activity_fraction"])
        if not 0.0 < activity_fraction < 1.0:
            raise ValueError("fairness.activity_fraction must be between 0 and 1")

        dataset = str(self.config["dataset"])
        partitions = metadata_partitions(
            profiles,
            activity_fraction,
            dataset,
        )
        partitions.update(
            latent_partitions(
                profiles=profiles,
                dataset=dataset,
                seed=int(self.config["seed"]),
                k_min=int(self.settings["cluster_k_min"]),
                k_max=int(self.settings["cluster_k_max"]),
                kmeans_n_init=int(self.settings["kmeans_n_init"]),
            )
        )

        global_ndcg = {
            str(k): float(np.mean([user.ndcg[str(k)] for user in evaluations]))
            for k in topk
        }
        self._validate_recbole_ndcg(global_ndcg, recbole_metrics)

        analysis = {
            "evaluation": {
                "dataset": str(self.config["dataset"]),
                "seed": int(self.config["seed"]),
                "test_users": len(evaluations),
                "test_interactions": sum(
                    len(user.squared_errors) for user in evaluations
                ),
                "ndcg": global_ndcg,
            },
            "groupings": {
                name: self._partition_metrics(partition, evaluations, topk)
                for name, partition in partitions.items()
            },
        }

        store = ResultsStore(self.settings["output_dir"])
        output_path = store.update(
            dataset=str(self.config["dataset"]),
            algorithm=self.algorithm,
            analysis=analysis,
        )
        return analysis, output_path

    def _evaluate_users(self, trainer, test_data, topk: list[int]):
        dataset = test_data._dataset
        uid_field = self.config["USER_ID_FIELD"]
        iid_field = self.config["ITEM_ID_FIELD"]
        rating_field = self.config["RATING_FIELD"]
        rating_min = float(self.settings["rating_min"])
        rating_max = float(self.settings["rating_max"])
        if rating_min >= rating_max:
            raise ValueError("fairness rating_min must be smaller than rating_max")

        test_ratings = {}
        for uid, iid, rating in zip(
            dataset.inter_feat[uid_field].tolist(),
            dataset.inter_feat[iid_field].tolist(),
            dataset.inter_feat[rating_field].tolist(),
        ):
            test_ratings[(int(uid), int(iid))] = float(rating)

        trainer.model.eval()
        trainer.tot_item_num = dataset.item_num
        if trainer.item_tensor is None:
            trainer.item_tensor = dataset.get_item_feature().to(trainer.device)

        discounts = 1.0 / np.log2(np.arange(max(topk), dtype=float) + 2.0)
        evaluations = []
        with torch.no_grad():
            for batched_data in test_data:
                interaction, scores, positive_u, positive_i = (
                    trainer._full_sort_batch_eval(batched_data)
                )
                top_items = torch.topk(scores, max(topk), dim=1).indices.cpu().numpy()
                batch_users = interaction[uid_field].detach().cpu().numpy()
                positive_rows = positive_u.detach().cpu().numpy()
                positive_items = positive_i.detach().cpu().numpy()
                score_values = scores.detach().cpu()

                for row, internal_uid in enumerate(batch_users):
                    row_items = positive_items[positive_rows == row]
                    if not len(row_items):
                        raise ValueError(f"Test user {internal_uid} has no positive items")

                    hits = np.isin(top_items[row], row_items)
                    ndcg = {}
                    for k in topk:
                        dcg = float(np.sum(discounts[:k] * hits[:k]))
                        ideal_length = min(len(row_items), k)
                        idcg = float(np.sum(discounts[:ideal_length]))
                        ndcg[str(k)] = dcg / idcg

                    squared_errors = []
                    for internal_iid in row_items:
                        actual = test_ratings[(int(internal_uid), int(internal_iid))]
                        predicted = float(score_values[row, int(internal_iid)])
                        if not math.isfinite(predicted):
                            raise ValueError(
                                "Model produced a non-finite prediction for a test pair"
                            )
                        clipped = min(rating_max, max(rating_min, predicted))
                        squared_errors.append((clipped - actual) ** 2)

                    user_id = str(dataset.id2token(uid_field, int(internal_uid)))
                    evaluations.append(
                        UserEvaluation(
                            user_id=user_id,
                            ndcg=ndcg,
                            squared_errors=tuple(squared_errors),
                        )
                    )

        evaluations.sort(key=lambda evaluation: evaluation.user_id)
        if not evaluations:
            raise ValueError("Final test evaluation contains no users")
        return evaluations

    def _development_activity(self, data_loaders: Iterable) -> Counter[str]:
        counts: Counter[str] = Counter()
        uid_field = self.config["USER_ID_FIELD"]
        for data_loader in data_loaders:
            if data_loader is None:
                continue
            dataset = data_loader._dataset
            internal_ids = dataset.inter_feat[uid_field].detach().cpu().numpy()
            external_ids = dataset.id2token(uid_field, internal_ids)
            counts.update(str(user_id) for user_id in external_ids)
        return counts

    def _read_profiles(self) -> dict[str, dict[str, Any]]:
        profile_path = self.dataset_dir / f"{self.config['dataset']}.user"
        with profile_path.open(encoding="utf-8", newline="") as input_file:
            reader = csv.DictReader(input_file, delimiter="\t")
            dataset = str(self.config["dataset"]).casefold()
            if dataset == "lastfm":
                return self._read_lastfm_profiles(reader, profile_path)
            if dataset == "yelp":
                return self._read_yelp_profiles(reader, profile_path)

        raise ValueError(f"Unsupported fairness dataset: {self.config['dataset']}")

    @staticmethod
    def _read_lastfm_profiles(reader, profile_path):
        required_fields = {"user_id:token", "gender:token", "age:float"}
        if reader.fieldnames is None or not required_fields.issubset(
            reader.fieldnames
        ):
            raise ValueError(f"Missing LastFM profile fields in {profile_path}")

        profiles = {}
        for row in reader:
            user_id = row["user_id:token"]
            raw_age = row["age:float"].strip()
            profiles[user_id] = {
                "gender": row["gender:token"],
                "age": float(raw_age) if raw_age else None,
            }
        return profiles

    @staticmethod
    def _read_yelp_profiles(reader, profile_path):
        required_fields = {
            "user_id:token",
            "is_active:token",
            "friend_count:float",
            "fans:float",
            "tenure_years:float",
        }
        if reader.fieldnames is None or not required_fields.issubset(
            reader.fieldnames
        ):
            raise ValueError(f"Missing Yelp profile fields in {profile_path}")

        profiles = {}
        for row in reader:
            user_id = row["user_id:token"]
            raw_is_active = row["is_active:token"].strip().casefold()
            if raw_is_active not in {"true", "false"}:
                raise ValueError(
                    f"Invalid is_active value for Yelp user {user_id}"
                )

            friend_count = float(row["friend_count:float"])
            fans = float(row["fans:float"])
            tenure_years = float(row["tenure_years:float"])
            if (
                not math.isfinite(friend_count)
                or friend_count < 0
                or not math.isfinite(fans)
                or fans < 0
                or not math.isfinite(tenure_years)
                or tenure_years < 0
            ):
                raise ValueError(f"Invalid Yelp profile values for user {user_id}")

            profiles[user_id] = {
                "is_active": raw_is_active == "true",
                "friend_count": friend_count,
                "fans": fans,
                "tenure_years": tenure_years,
            }
        return profiles

    @staticmethod
    def _partition_metrics(
        partition: Partition,
        evaluations: list[UserEvaluation],
        topk: list[int],
    ) -> dict[str, Any]:
        grouped: dict[str, list[UserEvaluation]] = {}
        for user in evaluations:
            try:
                group = partition.assignments[user.user_id]
            except KeyError as error:
                raise ValueError(f"User {user.user_id} has no group") from error
            grouped.setdefault(group, []).append(user)

        groups = {}
        losses = []
        for group, users in sorted(grouped.items()):
            errors = [error for user in users for error in user.squared_errors]
            if not errors:
                raise ValueError(f"Group {group} has no test interactions")
            loss = float(np.mean(errors))
            losses.append(loss)
            groups[group] = {
                "users": len(users),
                "test_interactions": len(errors),
                "mse": loss,
                "ndcg": {
                    str(k): float(np.mean([user.ndcg[str(k)] for user in users]))
                    for k in topk
                },
            }

        result = {**partition.metadata, "groups": groups}
        if len(losses) < 2:
            result.update(
                {
                    "valid": False,
                    "rgrp": None,
                    "reason": "partition has fewer than two non-empty groups",
                }
            )
            return result

        pairwise_variation = sum(
            (losses[first] - losses[second]) ** 2
            for first in range(len(losses))
            for second in range(first + 1, len(losses))
        )
        result.update(
            {
                "valid": True,
                "rgrp": float(pairwise_variation / (len(losses) ** 2)),
            }
        )
        return result

    @staticmethod
    def _validate_recbole_ndcg(
        detailed_ndcg: dict[str, float], recbole_metrics: dict[str, Any]
    ) -> None:
        for k, value in detailed_ndcg.items():
            metric_name = f"ndcg@{k}"
            if metric_name not in recbole_metrics:
                continue
            difference = abs(value - float(recbole_metrics[metric_name]))
            if difference > 5e-5:
                raise ValueError(
                    f"Detailed {metric_name}={value:.8f} differs from "
                    f"RecBole={float(recbole_metrics[metric_name]):.8f}"
                )
