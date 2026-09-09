import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

NORTH_AMERICA_COUNTRIES = {
    "Canada",
    "United States",
}

EUROPE_COUNTRIES = {
    "Albania", "Andorra", "Austria", "Belarus",
    "Belgium", "Bosnia and Herzegovina", "Bulgaria", "Croatia",
    "Cyprus", "Czech Republic", "Denmark", "Estonia",
    "Faroe Islands", "Finland", "France", "Germany",
    "Gibraltar", "Greece", "Holy See (Vatican City State)", "Hungary",
    "Iceland", "Ireland", "Italy", "Latvia", "Liechtenstein",
    "Lithuania", "Luxembourg", "Macedonia", "Malta",
    "Moldova", "Monaco", "Montenegro", "Netherlands",
    "Norway", "Poland", "Portugal", "Romania",
    "Russian Federation", "San Marino", "Serbia",
    "Slovakia", "Slovenia", "Spain", "Svalbard and Jan Mayen",
    "Sweden", "Switzerland", "Ukraine", "United Kingdom",
}


@dataclass(frozen=True)
class UserProfile:
    user_id: str
    development_interactions: int
    gender: str = ""
    age: float | None = None
    country: str = ""
    is_active: bool | None = None
    friend_count: float | None = None
    fans: float | None = None
    tenure_years: float | None = None


@dataclass(frozen=True)
class Partition:
    assignments: dict[str, str]
    metadata: dict[str, Any]


def metadata_partitions(profiles: list[UserProfile], activity_fraction, dataset):
    """
    Create metadata partitions for the given dataset.

    Args:
        profiles (list[UserProfile]): User profiles containing metadata for partitioning.
        activity_fraction: The fraction of active users to include in the partition.
        dataset: The dataset to use for partitioning.

    Returns:
        partitions: A dictionary containing the metadata partitions for the given dataset.
    """
    if dataset.casefold() == "yelp":
        return _yelp_metadata_partitions(profiles)

    gender = {
        profile.user_id: _gender_group(profile.gender) for profile in profiles
    }
    age = {profile.user_id: _age_group(profile.age) for profile in profiles}
    location = {
        profile.user_id: _location_group(profile.country) for profile in profiles
    }

    active_count = max(1, math.ceil(len(profiles) * activity_fraction))
    activity_order = sorted(
        profiles,
        key=lambda profile: (-profile.development_interactions, profile.user_id),
    )
    
    active_users = {
        profile.user_id for profile in activity_order[:active_count]
    }
    
    activity = {
        profile.user_id: (
            "active" if profile.user_id in active_users else "inactive"
        )
        for profile in profiles
    }

    return {
        "gender": Partition(gender, {"type": "metadata"}),
        "age": Partition(age, {"type": "metadata"}),
        "location": Partition(location, {"type": "metadata"}),
        "activity": Partition(
            activity,
            {
                "type": "metadata",
                "active_fraction": activity_fraction,
                "active_users": active_count,
            },
        ),
    }


def _yelp_metadata_partitions(profiles: list[UserProfile]):
    """
    Create metadata partitions for the Yelp dataset.

    Args:
        profiles (list[UserProfile]): User profiles.

    Returns:
        partitions: A dictionary containing the metadata partitions for the Yelp dataset. 
    """
    activity = {
        profile.user_id: "active" if profile.is_active else "inactive"
        for profile in profiles
    }
    
    friend_count = {
        profile.user_id: _friend_count_group(profile.friend_count)
        for profile in profiles
    }
    
    fans = {
        profile.user_id: _fans_group(profile.fans)
        for profile in profiles
    }
    
    tenure = {
        profile.user_id: _tenure_group(profile.tenure_years)
        for profile in profiles
    }

    return {
        "activity": Partition(
            activity,
            {
                "type": "metadata",
                "active_users": sum(
                    1 for profile in profiles if profile.is_active
                ),
            },
        ),
        "friend_count": Partition(friend_count, {"type": "metadata"}),
        "fans": Partition(fans, {"type": "metadata"}),
        "tenure": Partition(tenure, {"type": "metadata"}),
    }


def latent_partitions(profiles: list[UserProfile], dataset, seed, k_min, k_max, kmeans_n_init):
    """
    Perform latent variable clustering on the given user profiles.

    Args:
        profiles (list[UserProfile]): The user profiles to cluster.
        dataset: The dataset to use for clustering.
        seed: The random seed to use.
        k_min: The minimum number of clusters to consider.
        k_max: The maximum number of clusters to consider.
        kmeans_n_init: The number of times to run the k-means algorithm.

    Returns:
        partitions: A dictionary containing the clustering results for each algorithm. 
    """
    matrix = _feature_matrix(profiles, dataset)
    
    user_ids = [profile.user_id for profile in profiles]
    candidate_ks = range(k_min, min(k_max, len(profiles) - 1) + 1)

    algorithms = {
        "kmeans": lambda k: KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=kmeans_n_init,
            algorithm="lloyd",
            random_state=seed,
        ).fit_predict(matrix),
        "agglomerative": lambda k: AgglomerativeClustering(
            n_clusters=k,
            metric="euclidean",
            linkage="ward",
        ).fit_predict(matrix),
    }

    partitions = {}
    for name, cluster in algorithms.items():
        labels_by_k: dict[int, np.ndarray] = {}
        scores: dict[int, float] = {}
        
        for k in candidate_ks:
            labels = np.asarray(cluster(k), dtype=int)
            if len(np.unique(labels)) != k:
                continue
            
            score = float(silhouette_score(matrix, labels, metric="euclidean"))
            if not math.isfinite(score):
                continue
            
            labels_by_k[k] = labels
            scores[k] = score

        if not scores:
            raise ValueError(f"No valid {name} clustering for k={k_min}..{k_max}")

        selected_k = min(scores, key=lambda k: (-scores[k], k))
        labels = _canonical_labels(matrix, labels_by_k[selected_k], user_ids)
        assignments = dict(zip(user_ids, labels))
        partitions[name] = Partition(
            assignments,
            {
                "type": "latent",
                "selected_k": selected_k,
                "selected_silhouette": scores[selected_k],
                "silhouette_by_k": {
                    str(k): scores[k] for k in sorted(scores)
                },
            },
        )

    return partitions


def _feature_matrix(profiles: list[UserProfile], dataset: str) -> np.ndarray:
    """
    Create a feature matrix for the specified dataset.

    Args:
        profiles (list[UserProfile]): User profiles containing features for clustering.
        dataset (str): Dataset name.

    Returns:
        feature_matrix: Feature matrix for the specified dataset.
    """
    if dataset.casefold() == "yelp":
        return _yelp_feature_matrix(profiles)

    known_ages = [profile.age for profile in profiles if profile.age is not None]
    median_age = float(np.median(known_ages)) if known_ages else 0.0

    numeric = np.asarray([
        [
            profile.age if profile.age is not None else median_age,
            math.log1p(profile.development_interactions),
        ]
        for profile in profiles
    ], dtype=float)
    
    numeric = StandardScaler().fit_transform(numeric)

    missing_age = np.asarray(
        [[1.0 if profile.age is None else 0.0] for profile in profiles]
    )
    
    gender_names = ("male", "female", "unknown")
    gender = np.asarray([
        [float(_gender_group(profile.gender) == name) for name in gender_names]
        for profile in profiles
    ])

    location_names = ("north_america", "europe", "other_locations")
    location = np.asarray([
        [float(_location_group(profile.country) == name) for name in location_names]
        for profile in profiles
    ])
    
    return np.concatenate((numeric, missing_age, gender, location), axis=1)

def _yelp_feature_matrix(profiles: list[UserProfile]) -> np.ndarray:
    """
    Create a feature matrix for the Yelp dataset.

    Args:
        profiles (list[UserProfile]): User profiles containing features for clustering.

    Returns:
        feature_matrix: Matrix of features for clustering. Each line is a user profile, and each column is a feature.
    """
    rows = []
    
    for profile in profiles:
        rows.append([
            float(profile.is_active),
            math.log1p(profile.friend_count),
            math.log1p(profile.fans),
            profile.tenure_years,
            math.log1p(profile.development_interactions),
        ])

    return StandardScaler().fit_transform(np.asarray(rows, dtype=float))


def _canonical_labels(matrix: np.ndarray, labels: np.ndarray, user_ids: list[str]):
    """
    Convert the cluster labels to a canonical form.

    Args:
        matrix (np.ndarray): 
        labels (np.ndarray): 
        user_ids (list[str]): 

    Returns:
        _type_: 
    """
    ordering = []
    
    for label in np.unique(labels):
        indexes = np.flatnonzero(labels == label)
        centroid = tuple(np.mean(matrix[indexes], axis=0).tolist())
        first_user = min(user_ids[index] for index in indexes)
        
        ordering.append((centroid, first_user, int(label)))

    label_names = {
        label: f"group_{index}"
        for index, (_, _, label) in enumerate(sorted(ordering), start=1)
    }
    
    return [label_names[int(label)] for label in labels]


def _gender_group(value: str) -> str:
    normalized = value.strip().casefold()
    
    if normalized in {"m", "male"}:
        return "male"
    if normalized in {"f", "female"}:
        return "female"
    
    return "unknown"


def _age_group(age: float | None) -> str:
    if age is None:
        return "unknown"
    if age < 18:
        return "under_18"
    if age <= 24:
        return "18_24"
    if age <= 34:
        return "25_34"
    if age <= 44:
        return "35_44"
    if age <= 49:
        return "45_49"
    if age <= 55:
        return "50_55"
    
    return "over_55"


def _location_group(country: str) -> str:
    if country in NORTH_AMERICA_COUNTRIES:
        return "north_america"
    if country in EUROPE_COUNTRIES:
        return "europe"

    return "other_locations"


def _friend_count_group(friend_count: float) -> str:
    if friend_count == 0:
        return "no_friends"
    if friend_count <= 10:
        return "1_10"
    if friend_count <= 100:
        return "11_100"
    
    return "101_plus"


def _fans_group(fans: float) -> str:
    if fans == 0:
        return "no_fans"
    if fans == 1:
        return "one_fan"
    if fans <= 13:
        return "2_13"
    
    return "14_plus"


def _tenure_group(tenure_years: float) -> str:
    if tenure_years < 1:
        return "under_1"
    if tenure_years < 3:
        return "1_3"
    if tenure_years < 5:
        return "3_5"
    
    return "5_plus"
