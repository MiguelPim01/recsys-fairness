import csv
import json
import math
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.utils.console import styled_tqdm

# ----- Config
STATISTICS = [
    "raw_user_profiles",
    "complete_user_profiles",
    "dropped_incomplete_user_profiles",
    "raw_interactions",
    "dropped_missing_user_interactions",
    "dropped_incomplete_user_interactions",
    "dropped_missing_item_interactions",
    "dropped_duplicate_interactions",
    "interactions",
    "users",
    "active_users",
    "items",
]
# -----


class YelpTransformDataset:
    """Transform the Yelp Open Dataset into RecBole atomic files."""

    REVIEWS_FILENAME = "yelp_academic_dataset_review.json"
    TIPS_FILENAME = "yelp_academic_dataset_tip.json"
    USERS_FILENAME = "yelp_academic_dataset_user.json"
    BUSINESSES_FILENAME = "yelp_academic_dataset_business.json"
    DAYS_PER_YEAR = 365.2425
    RESTAURANT_CATEGORIES = {  # noqa: RUF012
        "restaurant",
        "restaurants",
        "food",
        "coffee",
        "tea",
        "coffee & tea",
        "bakery",
        "bakeries",
        "pizza",
        "burger",
        "burgers",
        "mexican",
        "italian",
        "chinese",
        "japanese",
        "sushi",
        "sushi bars",
        "breakfast",
        "brunch",
        "breakfast & brunch",
        "sandwich",
        "sandwiches",
        "barbeque",
        "barbecue",
        "bbq",
        "seafood",
        "salad",
        "dessert",
        "desserts",
        "ice cream",
        "ice cream & frozen yogurt",
        "cafe",
        "cafes",
    }

    def __init__(self, raw_dir, output_dir, use_restaurants_users_only=False):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.use_restaurants_users_only = use_restaurants_users_only

    def transform(self):
        """
        Transforms the Yelp dataset into RecBole format.

        Returns:
            statistics (dict[str, int]): Statistics dictionary.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        statistics = defaultdict(int)
        for name in STATISTICS:
            statistics[name] = 0

        activity_threshold, known_users, complete_users, total_users = self._activity_data(statistics)
        known_items = self._item_ids()

        restaurant_preference_users = None
        if self.use_restaurants_users_only:
            restaurant_items = self._restaurant_item_ids()
            restaurant_preference_users = self._restaurant_preference_users(
                complete_users,
                known_items,
                restaurant_items,
            )
            statistics["restaurant_preference_users"] = len(restaurant_preference_users)
        
        # Transforms interactions
        users, items, reference_date = self._transform_interactions(
            known_users,
            complete_users,
            known_items,
            restaurant_preference_users,
            statistics,
        )
        
        # Transforms users
        self._transform_users(
            users,
            reference_date,
            activity_threshold,
            total_users,
            statistics,
        )
        
        # Transforms items
        self._transform_items(items, len(known_items), statistics)

        statistics["activity_threshold"] = activity_threshold
        statistics["reference_date"] = reference_date.isoformat(sep=" ")
        
        return dict(statistics)

    def _transform_interactions(self, known_users, complete_users, known_items, selected_users, statistics):
        """
        Writes down the interaction matrix.
            - user_id
            - item_id
            - rating
            - timestamp

        Args:
            known_users (set[str]): The set of all user IDs in the dataset.
            complete_users (set[str]): User IDs with complete modeled metadata.
            known_items (set[str]): The set of all item IDs in the dataset.
            selected_users (set[str] | None): User IDs selected by preference, if enabled.
            statistics (defaultdict): The statistics dictionary.

        Returns:
            users: Dataset users ids.
            items: Dataset items ids.
            reference_date: The latest date in the dataset.
        """
        input_path = self.raw_dir / self.REVIEWS_FILENAME
        output_path = self.output_dir / "yelp.inter"
        total_interactions = self._line_count(input_path)
        
        users = set()
        items = set()
        interactions = {}
        
        reference_date: datetime | None = None

        with input_path.open(encoding="utf-8") as input_file, output_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.writer(output_file, delimiter="\t", lineterminator="\n")
            
            # Writes header
            writer.writerow([
                "user_id:token", 
                "item_id:token", 
                "rating:float", 
                "timestamp:float"
            ])

            progress = styled_tqdm(
                input_file,
                total=total_interactions,
                desc="  interactions",
                unit="interaction",
                dynamic_ncols=True,
            )
            for line in progress:
                review = json.loads(line)
                review_date = datetime.fromisoformat(review["date"])
                
                statistics["raw_interactions"] += 1
                
                if reference_date is None or review_date > reference_date:
                    reference_date = review_date

                if review["user_id"] not in known_users:
                    statistics["dropped_missing_user_interactions"] += 1
                    continue

                if review["user_id"] not in complete_users:
                    statistics["dropped_incomplete_user_interactions"] += 1
                    continue
                
                if review["business_id"] not in known_items:
                    statistics["dropped_missing_item_interactions"] += 1
                    continue

                if selected_users is not None and review["user_id"] not in selected_users:
                    continue

                interaction_id = (review["user_id"], review["business_id"])
                timestamp = review_date.replace(tzinfo=timezone.utc).timestamp()

                if interaction_id in interactions:
                    statistics["dropped_duplicate_interactions"] += 1

                if interaction_id not in interactions or timestamp > interactions[interaction_id][1]:
                    interactions[interaction_id] = (review["stars"], timestamp)

                users.add(review["user_id"])
                items.add(review["business_id"])

            for (user_id, item_id), (rating, timestamp) in interactions.items():
                writer.writerow([
                    user_id,
                    item_id,
                    rating,
                    timestamp,
                ])
                
                statistics["interactions"] += 1

        return users, items, reference_date

    def _activity_data(self, statistics):
        """
        Gathers the 95% most active users.

        Returns:
            review_count_threshold (int): The minimum number of reviews a user must have to be considered active.
            user_ids (set[str]): The set of all user IDs in the dataset.
            complete_user_ids (set[str]): User IDs with complete metadata.
            total_users (int): Total number of raw user profiles.
        """
        input_path = self.raw_dir / self.USERS_FILENAME
        
        review_counts = []
        user_ids: set[str] = set()
        complete_user_ids: set[str] = set()

        with input_path.open(encoding="utf-8") as input_file:
            for line in input_file:
                user = json.loads(line)

                statistics["raw_user_profiles"] += 1

                user_id = str(user.get("user_id") or "").strip()
                
                if not user_id:
                    statistics["dropped_incomplete_user_profiles"] += 1
                    continue
                
                if user_id in user_ids:
                    raise ValueError(f"Duplicate user profile for {user_id}")

                user_ids.add(user_id)

                if not self._has_complete_metadata(user):
                    statistics["dropped_incomplete_user_profiles"] += 1
                    continue

                review_counts.append(int(user["review_count"]))
                
                complete_user_ids.add(user_id)
                statistics["complete_user_profiles"] += 1

        review_counts.sort()
        percentile_index = math.ceil(0.95 * len(review_counts)) - 1
        
        return (
            review_counts[percentile_index],
            user_ids,
            complete_user_ids,
            statistics["raw_user_profiles"],
        )

    def _item_ids(self):
        """
        Gathers the IDs of all businesses in the dataset.

        Returns:
            item_ids (set[str]): The set of all business IDs in the dataset.
        """
        input_path = self.raw_dir / self.BUSINESSES_FILENAME
        item_ids = set()

        with input_path.open(encoding="utf-8") as input_file:
            for line in input_file:
                item_ids.add(json.loads(line)["business_id"])
        
        return item_ids

    def _restaurant_item_ids(self):
        """
        Gathers the IDs of businesses categorized as restaurants or food.

        Returns:
            item_ids (set[str]): Restaurant and food business IDs.
        """
        input_path = self.raw_dir / self.BUSINESSES_FILENAME
        item_ids = set()

        with input_path.open(encoding="utf-8") as input_file:
            for line in input_file:
                business = json.loads(line)

                if self._is_restaurant_business(business.get("categories")):
                    item_ids.add(business["business_id"])

        return item_ids

    def _restaurant_preference_users(self, complete_users, known_items, restaurant_items):
        """
        Selects users whose predominant preference is restaurants or food.

        Args:
            complete_users (set[str]): User IDs with complete modeled metadata.
            known_items (set[str]): The set of all business IDs in the dataset.
            restaurant_items (set[str]): Restaurant and food business IDs.

        Returns:
            users (set[str]): Users with more restaurant than non-restaurant interactions.
        """
        interaction_counts = defaultdict(lambda: [0, 0])

        for filename, description in (
            (self.REVIEWS_FILENAME, "  restaurant preference reviews"),
            (self.TIPS_FILENAME, "  restaurant preference tips"),
        ):
            input_path = self.raw_dir / filename
            total_interactions = self._line_count(input_path)

            with input_path.open(encoding="utf-8") as input_file:
                progress = styled_tqdm(
                    input_file,
                    total=total_interactions,
                    desc=description,
                    unit="interaction",
                    dynamic_ncols=True,
                )
                for line in progress:
                    interaction = json.loads(line)
                    user_id = interaction["user_id"]
                    item_id = interaction["business_id"]

                    if user_id not in complete_users or item_id not in known_items:
                        continue

                    interaction_type = 0 if item_id in restaurant_items else 1
                    interaction_counts[user_id][interaction_type] += 1

        return {
            user_id
            for user_id, (restaurant_count, other_count) in interaction_counts.items()
            if restaurant_count > other_count
        }

    def _transform_users(self, interaction_users, reference_date, activity_threshold, total_users, statistics):
        """
        Writes down the user metadata.
            - user_id
            - is_active
            - friend_count
            - fans
            - tenure_years

        Args:
            interaction_users (set[str]): The set of all user IDs in the dataset.
            reference_date (datetime): The latest date in the dataset.
            activity_threshold (int): The minimum number of reviews a user must have to be considered active.
            total_users (int): The total number of users in the dataset.
            statistics (defaultdict): The statistics dictionary.
        """
        input_path = self.raw_dir / self.USERS_FILENAME
        output_path = self.output_dir / "yelp.user"
        
        written_users = set()

        with input_path.open(encoding="utf-8") as input_file, output_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.writer(output_file, delimiter="\t", lineterminator="\n")
            
            # Writes header
            writer.writerow([
                "user_id:token",
                "is_active:token",
                "friend_count:float",
                "fans:float",
                "tenure_years:float",
            ])

            progress = styled_tqdm(
                input_file,
                total=total_users,
                desc="  users",
                unit="user",
                dynamic_ncols=True,
            )
            for line in progress:
                user = json.loads(line)
                user_id = user["user_id"]
                
                if user_id not in interaction_users:
                    continue

                review_count = int(user["review_count"])
                tenure = self._tenure_years(user["yelping_since"], reference_date)
                is_active = "true" if review_count >= activity_threshold else "false"
                
                writer.writerow([
                    user_id, 
                    is_active, 
                    self._friend_count(user["friends"]),
                    int(user["fans"]),
                    tenure
                ])

                written_users.add(user_id)
                statistics["users"] += 1
                
                if is_active == "true":
                    statistics["active_users"] += 1

        missing_users = interaction_users - written_users
        if missing_users:
            raise ValueError(f"Missing metadata for {len(missing_users)} Yelp users")

    def _transform_items(self, interaction_items, total_items, statistics):
        """
        Writes down item metadata.
            - item_id
            - business_name
            - city
            - state
            - postal_code
            - latitude
            - longitude
            - stars
            - review_count
            - is_open
            - categories
        
        Args:
            interaction_items (set[str]): The set of all item IDs in the dataset.
            total_items (int): The total number of items in the dataset.
            statistics (defaultdict): The statistics dictionary.
        """
        input_path = self.raw_dir / self.BUSINESSES_FILENAME
        output_path = self.output_dir / "yelp.item"
        
        written_items = set()

        with input_path.open(encoding="utf-8") as input_file, output_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.writer(output_file, delimiter="\t", lineterminator="\n")
            
            writer.writerow([
                "item_id:token",
                "business_name:token",
                "city:token",
                "state:token",
                "postal_code:token",
                "latitude:float",
                "longitude:float",
                "stars:float",
                "review_count:float",
                "is_open:token",
                "categories:token_seq",
            ])

            progress = styled_tqdm(
                input_file,
                total=total_items,
                desc="  items",
                unit="item",
                dynamic_ncols=True,
            )
            for line in progress:
                business = json.loads(line)
                item_id = business["business_id"]
                
                if item_id not in interaction_items:
                    continue

                writer.writerow([
                    item_id,
                    self._clean_token(business["name"]),
                    self._clean_token(business["city"]),
                    self._clean_token(business["state"]),
                    self._clean_token(business["postal_code"]),
                    business["latitude"],
                    business["longitude"],
                    business["stars"],
                    business["review_count"],
                    "true" if business["is_open"] else "false",
                    self._categories(business["categories"]),
                ])
                
                written_items.add(item_id)
                statistics["items"] += 1

        missing_items = interaction_items - written_items
        if missing_items:
            raise ValueError(f"Missing metadata for {len(missing_items)} Yelp businesses")

    @staticmethod
    def _friend_count(friends) -> int:
        if not friends or friends == "None":
            return 0
        
        if isinstance(friends, list):
            return len(friends)
        
        return friends.count(",") + 1

    @staticmethod
    def _has_complete_metadata(user) -> bool:
        user_id = str(user.get("user_id") or "").strip()
        yelping_since = str(user.get("yelping_since") or "").strip()

        if not user_id or not yelping_since or "friends" not in user:
            return False

        try:
            review_count = int(user["review_count"])
            fans = int(user["fans"])
            datetime.fromisoformat(yelping_since)
        except (KeyError, TypeError, ValueError):
            return False

        return review_count >= 0 and fans >= 0

    @staticmethod
    def _line_count(input_path) -> int:
        with input_path.open(encoding="utf-8") as input_file:
            return sum(1 for _ in input_file)

    @classmethod
    def _tenure_years(cls, yelping_since: str, reference_date: datetime) -> float:
        registration_date = datetime.fromisoformat(yelping_since)
        elapsed_days = (reference_date - registration_date).total_seconds() / 86400
        
        return round(max(0.0, elapsed_days / cls.DAYS_PER_YEAR), 6)

    @staticmethod
    def _clean_token(value) -> str:
        return " ".join(str(value or "").split())

    @classmethod
    def _categories(cls, categories) -> str:
        if not categories:
            return ""
        
        return " ".join(
            "_".join(cls._clean_token(category).split())
            for category in categories.split(",")
        )

    @classmethod
    def _is_restaurant_business(cls, categories) -> bool:
        if not categories:
            return False

        return any(
            cls._normalized_category(category) in cls.RESTAURANT_CATEGORIES
            for category in categories.split(",")
        )

    @classmethod
    def _normalized_category(cls, category) -> str:
        normalized = unicodedata.normalize(
            "NFKD",
            cls._clean_token(category).casefold(),
        )

        return "".join(
            character
            for character in normalized
            if not unicodedata.combining(character)
        )
