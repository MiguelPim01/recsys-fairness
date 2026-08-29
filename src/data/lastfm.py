import csv
import unicodedata
from collections import defaultdict
from pathlib import Path

from tqdm.auto import tqdm


class LastFMTransformDataset:
    """Transform the LastFM-360K files into RecBole atomic files."""

    INTERACTIONS_FILENAME = "usersha1-artmbid-artname-plays.tsv"
    PROFILES_FILENAME = "usersha1-profile.tsv"

    def __init__(self, raw_dir, output_dir):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)

    def transform(self):
        """
        Transforms the LastFM-360K dataset into RecBole format.

        Returns:
            statistics (dict[str, int]): Statistics dictionary.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        statistics = defaultdict(int)

        # Writing interaction matrix
        profile_users = self._read_profile_users()
        interaction_users, items = self._transform_interactions(profile_users, statistics)

        # Writing user and item profiles
        self._transform_users(interaction_users, len(profile_users), statistics)
        self._write_items(items)

        statistics["users"] = len(interaction_users)
        statistics["items"] = len(items)

        return dict(statistics)

    def _transform_interactions(self, profile_users, statistics):
        """
        Main logic for writing lastfm.inter file.

        Args:
            profile_users: Set of user IDs from the profile data.
            statistics: Statistics dictionary.

        Returns:
            users: Set of user IDs from the interaction data.
            items: Dictionary mapping item_id to (musicbrainz_artist_id, artist_name).
        """
        input_path = self.raw_dir / self.INTERACTIONS_FILENAME
        output_path = self.output_dir / "lastfm.inter"
        total_interactions = self._line_count(input_path)

        users = set()
        items = {}
        current_user = None
        current_interactions = {}

        with input_path.open(encoding="utf-8", newline="") as input_file, output_path.open("w", encoding="utf-8", newline="") as output_file:
            reader = csv.reader(input_file, delimiter="\t", quoting=csv.QUOTE_NONE)
            writer = csv.writer(output_file, delimiter="\t", lineterminator="\n")

            # Writing header
            writer.writerow([
                "user_id:token",
                "item_id:token",
                "play_count:float"
            ])

            progress = tqdm(reader, total=total_interactions, desc="  interactions", unit="interaction", dynamic_ncols=True)
            for row in progress:
                statistics["raw_interactions"] += 1

                if len(row) != 4:
                    raise ValueError(
                        f"Expected 4 columns at interaction row "
                        f"{statistics['raw_interactions']}"
                    )

                user_id, musicbrainz_id, artist_name, raw_play_count = map(str.strip, row)

                if not user_id:
                    statistics["dropped_invalid_user_rows"] += 1
                    continue
                if user_id not in profile_users:
                    statistics["dropped_missing_profile_user_rows"] += 1
                    continue

                play_count = int(raw_play_count)
                if play_count <= 0:
                    statistics["dropped_nonpositive_play_rows"] += 1
                    continue

                if not musicbrainz_id and not artist_name:
                    statistics["dropped_missing_artist_rows"] += 1
                    continue

                # IDs will be MBID or the artist name
                item_id = self._item_id(musicbrainz_id, artist_name)

                if user_id != current_user:
                    if current_user is not None:
                        self._write_user_interactions(writer, current_user, current_interactions, statistics)

                    if user_id in users:
                        raise ValueError(
                            "The interactions file is not grouped by user; "
                            f"user {user_id} appears more than once"
                        )

                    current_user = user_id
                    current_interactions = {}
                    users.add(user_id)

                if item_id in current_interactions:
                    current_interactions[item_id] += play_count
                    statistics["aggregated_duplicate_rows"] += 1
                else:
                    current_interactions[item_id] = play_count

                items.setdefault(item_id, (musicbrainz_id, artist_name))

            if current_user is not None:
                self._write_user_interactions(writer, current_user, current_interactions, statistics)

        return users, items

    def _read_profile_users(self):
        """
        Reads all users so we don't compute unnecessary interactions later on.

        Returns:
            users: Set of user IDs from the profile data.
        """
        input_path = self.raw_dir / self.PROFILES_FILENAME
        users = set()

        with input_path.open(encoding="utf-8", newline="") as input_file:
            reader = csv.reader(input_file, delimiter="\t", quoting=csv.QUOTE_NONE)

            for row_number, row in enumerate(reader, start=1):
                if len(row) != 5:
                    raise ValueError(
                        f"Expected 5 columns at profile row {row_number}"
                    )

                user_id = row[0].strip()
                users.add(user_id)

        return users

    def _transform_users(self, interaction_users, total_users, statistics):
        """
        Writes user data to lastfm.user file:
            - user_id
            - gender
            - age
            - country
            - signup_date

        Args:
            interaction_users: Set of user IDs from the interaction data.
            total_users: Total number of users in the profile data.
            statistics: Statistics dictionary.
        """
        input_path = self.raw_dir / self.PROFILES_FILENAME
        output_path = self.output_dir / "lastfm.user"
        written_users: set[str] = set()

        with input_path.open(encoding="utf-8", newline="") as input_file, output_path.open("w", encoding="utf-8", newline="") as output_file:
            reader = csv.reader(input_file, delimiter="\t", quoting=csv.QUOTE_NONE)
            writer = csv.writer(output_file, delimiter="\t", lineterminator="\n")

            # Writing header
            writer.writerow([
                "user_id:token",
                "gender:token",
                "age:float",
                "country:token",
                "signup_date:token",
            ])

            progress = tqdm(reader, total=total_users, desc="  users", unit="user", dynamic_ncols=True)
            for row_number, row in enumerate(progress, start=1):
                if len(row) != 5:
                    raise ValueError(
                        f"Expected 5 columns at profile row {row_number}"
                    )

                user_id, gender, raw_age, country, signup_date = map(str.strip, row)

                if user_id not in interaction_users:
                    continue
                if user_id in written_users:
                    raise ValueError(f"Duplicate user profile for {user_id}")

                age = self._valid_age(raw_age)
                if raw_age and not age:
                    statistics["invalid_ages"] += 1

                # Adding user data
                writer.writerow([
                    user_id,
                    gender,
                    age,
                    country,
                    signup_date,
                ])

                written_users.add(user_id)

        missing_count = len(interaction_users) - len(written_users)
        if missing_count:
            raise ValueError(f"Missing profiles for {missing_count} interaction users")

        statistics["written_user_profiles"] = len(written_users)

    def _write_items(self, items):
        """
        Writes item data to lastfm.item file:
            - item_id
            - musicbrainz_artist_id
            - artist_name

        Args:
            items: Dictionary mapping item_id to (musicbrainz_artist_id, artist_name).
        """
        output_path = self.output_dir / "lastfm.item"

        with output_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.writer(output_file, delimiter="\t", lineterminator="\n")

            # Writing header
            writer.writerow([
                "item_id:token",
                "musicbrainz_artist_id:token",
                "artist_name:token",
            ])

            # Writing item profiles
            progress = tqdm(items.items(), total=len(items), desc="  items", unit="item", dynamic_ncols=True)
            for item_id, (musicbrainz_id, artist_name) in progress:
                writer.writerow([item_id, musicbrainz_id, artist_name])

    def _item_id(cls, musicbrainz_id: str, artist_name: str) -> str:
        if musicbrainz_id:
            return musicbrainz_id

        normalized_name = " ".join(
            unicodedata.normalize("NFKC", artist_name).casefold().split()
        )

        if not normalized_name:
            raise ValueError("An artist without a MusicBrainz ID must have a name")

        return f"name:{normalized_name}"

    @staticmethod
    def _write_user_interactions(writer, user_id, interactions, statistics):
        """
        Writes interaction data to lastfm.inter file for a specific user:
            - user_id
            - item_id
            - play_count

        Args:
            writer: lastfm.inter writer.
            user_id: User ID.
            interactions: [item_id, play_count] pairs for the user.
            statistics: Statistics dictionary.
        """
        for item_id, play_count in interactions.items():
            writer.writerow([user_id, item_id, play_count])

            statistics["written_interactions"] += 1

    @staticmethod
    def _line_count(input_path) -> int:
        with input_path.open(encoding="utf-8", newline="") as input_file:
            return sum(1 for _ in input_file)

    @staticmethod
    def _valid_age(raw_age: str) -> str:
        if not raw_age.isdigit():
            return ""

        numeric_age = int(raw_age)

        return str(numeric_age) if numeric_age > 0 else ""
