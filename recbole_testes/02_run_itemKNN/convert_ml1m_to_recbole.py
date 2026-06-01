from pathlib import Path
import re
import pandas as pd


RAW_DIR = Path("./data/raw/ml-1m")
OUT_DIR = Path("./data/atomic/ml-1m")


def split_title_and_year(title: str):
    """
    MovieLens titles usually look like:
    Toy Story (1995)

    RecBole's ml-1m.item example separates:
    movie_title:token_seq
    release_year:token
    """
    match = re.search(r"\((\d{4})\)$", title)

    if match:
        year = match.group(1)
        clean_title = title[:match.start()].strip()
    else:
        year = "unknown"
        clean_title = title.strip()

    return clean_title, year


def convert_inter():
    ratings_path = RAW_DIR / "ratings.dat"

    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=["user_id", "item_id", "rating", "timestamp"],
        encoding="latin-1"
    )

    ratings = ratings[["user_id", "item_id", "rating", "timestamp"]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ratings.to_csv(
        OUT_DIR / "ml-1m.inter",
        sep="\t",
        index=False,
        header=["user_id:token", "item_id:token", "rating:float", "timestamp:float"]
    )


def convert_user():
    users_path = RAW_DIR / "users.dat"

    users = pd.read_csv(
        users_path,
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        encoding="latin-1"
    )

    users = users[["user_id", "age", "gender", "occupation", "zip_code"]]

    users.to_csv(
        OUT_DIR / "ml-1m.user",
        sep="\t",
        index=False,
        header=[
            "user_id:token",
            "age:token",
            "gender:token",
            "occupation:token",
            "zip_code:token"
        ]
    )


def convert_item():
    movies_path = RAW_DIR / "movies.dat"

    movies = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        names=["item_id", "raw_title", "genres"],
        encoding="latin-1"
    )

    movie_titles = []
    release_years = []
    genre_sequences = []

    for _, row in movies.iterrows():
        clean_title, year = split_title_and_year(row["raw_title"])

        movie_titles.append(clean_title)
        release_years.append(year)

        # Raw MovieLens genres are pipe-separated, e.g. Animation|Children's|Comedy.
        # RecBole token_seq normally uses spaces.
        genre_sequences.append(str(row["genres"]).replace("|", " "))

    items = pd.DataFrame({
        "item_id": movies["item_id"],
        "movie_title": movie_titles,
        "release_year": release_years,
        "genre": genre_sequences
    })

    items.to_csv(
        OUT_DIR / "ml-1m.item",
        sep="\t",
        index=False,
        header=[
            "item_id:token",
            "movie_title:token_seq",
            "release_year:token",
            "genre:token_seq"
        ]
    )


if __name__ == "__main__":
    convert_inter()
    convert_user()
    convert_item()

    print(f"Atomic files created in: {OUT_DIR.resolve()}")
    print("Created:")
    print(" - ml-1m.inter")
    print(" - ml-1m.user")
    print(" - ml-1m.item")