import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


class YelpTransformDataset:
    """Transform the Yelp Open Dataset into RecBole atomic files."""

    REVIEWS_FILENAME = "yelp_academic_dataset_review.json"
    USERS_FILENAME = "yelp_academic_dataset_user.json"
    BUSINESSES_FILENAME = "yelp_academic_dataset_business.json"
    DAYS_PER_YEAR = 365.2425

    def __init__(self, raw_dir: Path | str, output_dir: Path | str) -> None:
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)

    def transform(self) -> dict[str, int | str]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        statistics: defaultdict[str, int | str] = defaultdict(int)

        activity_threshold, known_users = self._activity_data()
        known_items = self._item_ids()
        users, items, reference_date = self._transform_interactions(
            known_users, known_items, statistics
        )
        self._transform_users(
            users, reference_date, activity_threshold, statistics
        )
        self._transform_items(items, statistics)

        statistics["activity_threshold"] = activity_threshold
        statistics["reference_date"] = reference_date.isoformat(sep=" ")
        return dict(statistics)

    def _transform_interactions(
        self,
        known_users: set[str],
        known_items: set[str],
        statistics: defaultdict[str, int | str],
    ) -> tuple[set[str], set[str], datetime]:
        input_path = self.raw_dir / self.REVIEWS_FILENAME
        output_path = self.output_dir / "yelp.inter"
        users: set[str] = set()
        items: set[str] = set()
        reference_date: datetime | None = None

        with input_path.open(encoding="utf-8") as input_file, output_path.open(
            "w", encoding="utf-8", newline=""
        ) as output_file:
            writer = csv.writer(output_file, delimiter="\t", lineterminator="\n")
            writer.writerow(
                ["user_id:token", "item_id:token", "rating:float", "timestamp:float"]
            )

            for line in input_file:
                review = json.loads(line)
                review_date = datetime.fromisoformat(review["date"])
                statistics["raw_interactions"] += 1
                if reference_date is None or review_date > reference_date:
                    reference_date = review_date

                if review["user_id"] not in known_users:
                    statistics["dropped_missing_user_interactions"] += 1
                    continue
                if review["business_id"] not in known_items:
                    statistics["dropped_missing_item_interactions"] += 1
                    continue

                writer.writerow(
                    [
                        review["user_id"],
                        review["business_id"],
                        review["stars"],
                        review_date.replace(tzinfo=timezone.utc).timestamp(),
                    ]
                )

                users.add(review["user_id"])
                items.add(review["business_id"])
                statistics["interactions"] += 1

        if reference_date is None:
            raise ValueError("The Yelp review dataset is empty")
        return users, items, reference_date

    def _activity_data(self) -> tuple[int, set[str]]:
        input_path = self.raw_dir / self.USERS_FILENAME
        review_counts = []
        user_ids: set[str] = set()

        with input_path.open(encoding="utf-8") as input_file:
            for line in input_file:
                user = json.loads(line)
                review_counts.append(int(user["review_count"]))
                user_ids.add(user["user_id"])

        if not review_counts:
            raise ValueError("The Yelp user dataset is empty")

        review_counts.sort()
        percentile_index = math.ceil(0.95 * len(review_counts)) - 1
        return review_counts[percentile_index], user_ids

    def _item_ids(self) -> set[str]:
        input_path = self.raw_dir / self.BUSINESSES_FILENAME
        item_ids: set[str] = set()

        with input_path.open(encoding="utf-8") as input_file:
            for line in input_file:
                item_ids.add(json.loads(line)["business_id"])
        return item_ids

    def _transform_users(
        self,
        interaction_users: set[str],
        reference_date: datetime,
        activity_threshold: int,
        statistics: defaultdict[str, int | str],
    ) -> None:
        input_path = self.raw_dir / self.USERS_FILENAME
        output_path = self.output_dir / "yelp.user"
        written_users: set[str] = set()

        with input_path.open(encoding="utf-8") as input_file, output_path.open(
            "w", encoding="utf-8", newline=""
        ) as output_file:
            writer = csv.writer(output_file, delimiter="\t", lineterminator="\n")
            writer.writerow(
                [
                    "user_id:token",
                    "is_active:token",
                    "friend_count:float",
                    "tenure_years:float",
                ]
            )

            for line in input_file:
                user = json.loads(line)
                user_id = user["user_id"]
                if user_id not in interaction_users:
                    continue

                review_count = int(user["review_count"])
                tenure = self._tenure_years(user["yelping_since"], reference_date)
                is_active = "true" if review_count >= activity_threshold else "false"
                writer.writerow(
                    [user_id, is_active, self._friend_count(user["friends"]), tenure]
                )

                written_users.add(user_id)
                statistics["users"] += 1
                if is_active == "true":
                    statistics["active_users"] += 1

        missing_users = interaction_users - written_users
        if missing_users:
            raise ValueError(f"Missing metadata for {len(missing_users)} Yelp users")

    def _transform_items(
        self,
        interaction_items: set[str],
        statistics: defaultdict[str, int | str],
    ) -> None:
        input_path = self.raw_dir / self.BUSINESSES_FILENAME
        output_path = self.output_dir / "yelp.item"
        written_items: set[str] = set()

        with input_path.open(encoding="utf-8") as input_file, output_path.open(
            "w", encoding="utf-8", newline=""
        ) as output_file:
            writer = csv.writer(output_file, delimiter="\t", lineterminator="\n")
            writer.writerow(
                [
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
                ]
            )

            for line in input_file:
                business = json.loads(line)
                item_id = business["business_id"]
                if item_id not in interaction_items:
                    continue

                writer.writerow(
                    [
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
                    ]
                )
                written_items.add(item_id)
                statistics["items"] += 1

        missing_items = interaction_items - written_items
        if missing_items:
            raise ValueError(f"Missing metadata for {len(missing_items)} Yelp businesses")

    @staticmethod
    def _friend_count(friends: str | list[str] | None) -> int:
        if not friends or friends == "None":
            return 0
        if isinstance(friends, list):
            return len(friends)
        return friends.count(",") + 1

    @classmethod
    def _tenure_years(cls, yelping_since: str, reference_date: datetime) -> float:
        registration_date = datetime.fromisoformat(yelping_since)
        elapsed_days = (reference_date - registration_date).total_seconds() / 86400
        return round(max(0.0, elapsed_days / cls.DAYS_PER_YEAR), 6)

    @staticmethod
    def _clean_token(value: object) -> str:
        return " ".join(str(value or "").split())

    @classmethod
    def _categories(cls, categories: str | None) -> str:
        if not categories:
            return ""
        return " ".join(
            "_".join(cls._clean_token(category).split())
            for category in categories.split(",")
        )
