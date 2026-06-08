import re
import pandas as pd
import streamlit as st

from utils.auth import init_auth, is_logged_in, current_user, logout
from utils.model_loader import require_bundle
from utils.user_store import get_user_ratings, compute_user_stats_from_ratings


def clean_movie_title(title):
    """Make MovieLens titles look natural in the interface.

    MovieLens often stores article titles as:
    - Matrix, The
    - Beautiful Mind, A
    - American Tail, An

    This function converts them to:
    - The Matrix
    - A Beautiful Mind
    - An American Tail
    """
    title = str(title).strip()
    title = re.sub(r"\s+", " ", title)

    if not title or title.lower() == "nan":
        return "Unknown title"

    # Keep the year at the end, if it exists.
    year = ""
    year_match = re.search(r"\s*(\(\d{4}\))\s*$", title)
    if year_match:
        year = year_match.group(1)
        title = title[: year_match.start()].strip()

    endings = {
        ", The": "The ",
        ", A": "A ",
        ", An": "An ",
    }

    for ending, prefix in endings.items():
        if title.endswith(ending):
            title = prefix + title[: -len(ending)].strip()
            break

    title = re.sub(r"\s+", " ", title).strip()
    return f"{title} {year}".strip() if year else title


def require_login():
    init_auth()
    if not is_logged_in():
        st.warning("Please log in first. The app only makes predictions for your own account.")
        st.page_link("Home.py", label="Go to Authentication", icon="🔐")
        st.stop()
    return current_user()


def user_header():
    user = current_user()
    if not user:
        return

    c1, c2 = st.columns([5, 1])
    c1.caption(f"👤 Logged in as **{user['username']}**")

    if c2.button("Log out", key=f"logout_{st.session_state.get('_page_key', 'page')}"):
        logout()
        st.rerun()


def load_movie_catalog():
    bundle = require_bundle()
    catalog = bundle.get("movie_catalog")

    if not isinstance(catalog, pd.DataFrame) or catalog.empty:
        st.error("Movie catalogue is missing from the exported model bundle.")
        st.stop()

    cols = [
        c
        for c in [
            "movieId",
            "title",
            "genres",
            "year",
            "movie_avg_rating",
            "movie_rating_count",
        ]
        if c in catalog.columns
    ]

    catalog = catalog[cols].drop_duplicates("movieId").copy()
    catalog["title"] = catalog["title"].fillna("Unknown title").astype(str).apply(clean_movie_title)

    return bundle, catalog


def get_logged_user_context():
    user = require_login()
    user_id = int(user["userId"])
    ratings = get_user_ratings(user_id)
    stats = compute_user_stats_from_ratings(ratings)
    return user, user_id, ratings, stats


def movie_selectbox(movie_catalog: pd.DataFrame, key: str, label="Choose a movie") -> int:
    """Movie picker with two options: search mode or full dropdown mode.

    The UI shows only movie titles. The internal movieId is returned silently.
    """
    options = movie_catalog[["movieId", "title"]].drop_duplicates("movieId").copy()
    options["title"] = options["title"].fillna("Unknown title").astype(str).apply(clean_movie_title)
    options = options.sort_values("title").reset_index(drop=True)

    selection_mode = st.radio(
        "How do you want to choose the movie?",
        ["Search by title", "Use full dropdown list"],
        horizontal=True,
        key=f"{key}_selection_mode",
    )

    if selection_mode == "Search by title":
        search_text = st.text_input(
            "Search movie title",
            key=f"{key}_search_text",
            placeholder="Example: Hellboy, Matrix, Titanic...",
        ).strip()

        if len(search_text) < 2:
            st.info("Type at least 2 letters, then choose a movie from the filtered dropdown.")
            filtered = options.head(0).copy()
        else:
            mask = options["title"].str.contains(search_text, case=False, na=False, regex=False)
            filtered = options.loc[mask].copy()

            if filtered.empty:
                st.warning("No movie titles matched your search. Try fewer words or use the full dropdown list.")
                st.stop()

            if len(filtered) > 100:
                st.caption(f"Found {len(filtered)} movies. Showing the first 100 matches. Type more letters to narrow the list.")
                filtered = filtered.head(100).copy()
            else:
                st.caption(f"Found {len(filtered)} matching movie(s).")

        if filtered.empty:
            st.stop()

        safe_search_key = str(abs(hash(search_text)))
        select_key = f"{key}_search_result_{safe_search_key}"

    else:
        filtered = options.copy()
        select_key = f"{key}_full_dropdown"

    records = list(filtered[["movieId", "title"]].itertuples(index=False, name=None))

    selected = st.selectbox(
        label,
        records,
        key=select_key,
        format_func=lambda item: item[1],
    )

    return int(selected[0])


def show_rating_summary(ratings: pd.DataFrame):
    if ratings.empty:
        st.info("You have not rated any movies yet. Add a few ratings to make the recommendations more personal.")
        return

    r = ratings["rating"].astype(float)

    c1, c2, c3 = st.columns(3)
    c1.metric("Movies rated", len(r))
    c2.metric("Average rating", f"{r.mean():.2f}")
    c3.metric("Rating range", f"{r.min():.1f} – {r.max():.1f}")
