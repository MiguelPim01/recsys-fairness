import csv
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

USER_ID_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class LastFMTransformDataset:
    """Transform the LastFM-360K files into RecBole atomic files."""

    INTERACTIONS_FILENAME = "usersha1-artmbid-artname-plays.tsv"
    PROFILES_FILENAME = "usersha1-profile.tsv"

    def __init__(self, raw_dir: Path | str, output_dir: Path | str) -> None:
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)

    def transform(self) -> dict[str, int]:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        statistics: defaultdict[str, int] = defaultdict(int)
        interaction_users, items = self._transform_interactions(statistics)
        self._transform_users(interaction_users, statistics)
        self._write_items(items)

        statistics["users"] = len(interaction_users)
        statistics["items"] = len(items)
        return dict(statistics)

    def _transform_interactions(
        self, statistics: defaultdict[str, int]
    ) -> tuple[set[str], dict[str, tuple[str, str]]]:
        input_path = self.raw_dir / self.INTERACTIONS_FILENAME
        output_path = self.output_dir / "lastfm.inter"

        users: set[str] = set()
        items: dict[str, tuple[str, str]] = {}
        current_user: str | None = None
        current_interactions: dict[str, int] = {}

        with input_path.open(encoding="utf-8", newline="") as input_file, output_path.open(
            "w", encoding="utf-8", newline=""
        ) as output_file:
            reader = csv.reader(input_file, delimiter="\t", quoting=csv.QUOTE_NONE)
            writer = csv.writer(output_file, delimiter="\t", lineterminator="\n")
            writer.writerow(
                ["user_id:token", "item_id:token", "play_count:float"]
            )

            for row in reader:
                statistics["raw_interactions"] += 1
                if len(row) != 4:
                    raise ValueError(
                        f"Expected 4 columns at interaction row "
                        f"{statistics['raw_interactions']}"
                    )

                user_id, musicbrainz_id, artist_name, raw_play_count = row
                user_id = user_id.strip()

                if USER_ID_PATTERN.fullmatch(user_id) is None:
                    statistics["dropped_invalid_user_rows"] += 1
                    continue

                play_count = int(raw_play_count.strip())
                if play_count <= 0:
                    statistics["dropped_nonpositive_play_rows"] += 1
                    continue

                musicbrainz_id = musicbrainz_id.strip()
                artist_name = artist_name.strip()
                if not musicbrainz_id and not artist_name:
                    statistics["dropped_missing_artist_rows"] += 1
                    continue
                item_id = self._item_id(musicbrainz_id, artist_name)

                if user_id != current_user:
                    if current_user is not None:
                        self._write_user_interactions(
                            writer, current_user, current_interactions, statistics
                        )
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
                self._write_user_interactions(
                    writer, current_user, current_interactions, statistics
                )

        return users, items

    @staticmethod
    def _write_user_interactions(
        writer: csv.writer,
        user_id: str,
        interactions: dict[str, int],
        statistics: defaultdict[str, int],
    ) -> None:
        for item_id, play_count in interactions.items():
            writer.writerow([user_id, item_id, play_count])
            statistics["written_interactions"] += 1

    def _transform_users(
        self, interaction_users: set[str], statistics: defaultdict[str, int]
    ) -> None:
        input_path = self.raw_dir / self.PROFILES_FILENAME
        output_path = self.output_dir / "lastfm.user"
        written_users: set[str] = set()

        with input_path.open(encoding="utf-8", newline="") as input_file, output_path.open(
            "w", encoding="utf-8", newline=""
        ) as output_file:
            reader = csv.reader(input_file, delimiter="\t", quoting=csv.QUOTE_NONE)
            writer = csv.writer(output_file, delimiter="\t", lineterminator="\n")
            writer.writerow(
                [
                    "user_id:token",
                    "gender:token",
                    "age:float",
                    "country:token",
                    "signup_date:token",
                ]
            )

            for row_number, row in enumerate(reader, start=1):
                if len(row) != 5:
                    raise ValueError(
                        f"Expected 5 columns at profile row {row_number}"
                    )

                user_id, gender, raw_age, country, signup_date = row
                user_id = user_id.strip()
                if user_id not in interaction_users:
                    continue
                if user_id in written_users:
                    raise ValueError(f"Duplicate user profile for {user_id}")

                age = self._valid_age(raw_age)
                if raw_age.strip() and not age:
                    statistics["invalid_ages"] += 1

                writer.writerow(
                    [
                        user_id,
                        gender.strip(),
                        age,
                        country.strip(),
                        signup_date.strip(),
                    ]
                )
                written_users.add(user_id)

        missing_profiles = interaction_users - written_users
        if missing_profiles:
            raise ValueError(
                f"Missing profiles for {len(missing_profiles)} interaction users"
            )

        statistics["written_user_profiles"] = len(written_users)

    def _write_items(self, items: dict[str, tuple[str, str]]) -> None:
        output_path = self.output_dir / "lastfm.item"

        with output_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.writer(output_file, delimiter="\t", lineterminator="\n")
            writer.writerow(
                [
                    "item_id:token",
                    "musicbrainz_artist_id:token",
                    "artist_name:token",
                ]
            )
            for item_id, (musicbrainz_id, artist_name) in items.items():
                writer.writerow([item_id, musicbrainz_id, artist_name])

    @classmethod
    def _item_id(cls, musicbrainz_id: str, artist_name: str) -> str:
        if musicbrainz_id:
            return musicbrainz_id

        normalized_name = cls._normalize_artist_name(artist_name)
        if not normalized_name:
            raise ValueError("An artist without a MusicBrainz ID must have a name")
        return f"name:{normalized_name}"

    @staticmethod
    def _normalize_artist_name(artist_name: str) -> str:
        normalized = unicodedata.normalize("NFKC", artist_name)
        return " ".join(normalized.casefold().split())

    @staticmethod
    def _valid_age(raw_age: str) -> str:
        age = raw_age.strip()
        if not age.isdigit():
            return ""

        numeric_age = int(age)
        return str(numeric_age) if 1 <= numeric_age <= 120 else ""
