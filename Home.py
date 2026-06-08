import pandas as pd
import streamlit as st

from utils.auth import init_auth, is_logged_in, current_user, logout, _login_form
from utils.page_helpers import load_movie_catalog, movie_selectbox, clean_movie_title
from utils.user_store import add_rating, delete_rating, get_user_ratings
from utils.recommender import (
    recommend_with_clustering,
    predict_for_single_movie,
)

st.set_page_config(page_title="Movies Recommender", page_icon="🎬", layout="wide")
st.session_state["_page_key"] = "home"
init_auth()

# -------------------------------------------------------------------
# Visual style
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --bg: #ffffff;
        --panel: #ffffff;
        --panel-soft: #f8fafc;
        --text: #000000;
        --muted: #4b5563;
        --line: #e5e7eb;
        --accent: #2563eb;
        --accent-soft: #eff6ff;
        --green: #059669;
        --amber: #f59e0b;
        --danger: #dc2626;
    }

    .stApp {
        background: #ffffff !important;
        color: var(--text) !important;
    }

    [data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.95) !important;
        border-bottom: 1px solid var(--line);
    }

    [data-testid="stToolbar"], [data-testid="stDecoration"] {
        background: transparent !important;
    }

    [data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] * {
        color: #000000 !important;
    }

    [data-testid="stSidebar"] .stRadio label {
        font-weight: 700;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label {
        width: 100%;
        justify-content: flex-start;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 0.78rem 0.9rem;
        background: #ffffff;
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        transition: all 0.15s ease-in-out;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        border-color: var(--accent);
        background: var(--accent-soft);
        transform: translateX(2px);
    }

    .block-container {
        max-width: 1220px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: #000000;
        letter-spacing: -0.01em;
    }

    h1, h2, h3 {
        font-weight: 850;
    }

    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding: 0.25rem 0 1.2rem 0;
    }

    .brand-title {
        font-size: 1.35rem;
        font-weight: 900;
        color: #000000;
    }

    .brand-caption {
        color: var(--muted);
        font-size: 0.94rem;
        margin-top: 0.15rem;
    }

    .hero-card, .section-card, .result-card, .metric-card {
        border: 1px solid var(--line);
        background: #ffffff;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
        border-radius: 22px;
    }

    .hero-card {
        padding: 2rem;
        margin-bottom: 1.5rem;
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    }

    .hero-title {
        font-size: 2.3rem;
        font-weight: 900;
        margin-bottom: 0.35rem;
        color: #000000;
    }

    .hero-subtitle {
        max-width: 850px;
        color: var(--muted);
        font-size: 1.06rem;
        line-height: 1.65;
    }

    .metric-card {
        padding: 1.25rem 1.35rem;
        min-height: 130px;
        background: #ffffff;
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.92rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        font-size: 2.15rem;
        font-weight: 900;
        color: #000000;
        line-height: 1.1;
    }

    .metric-help {
        color: var(--muted);
        font-size: 0.86rem;
        margin-top: 0.55rem;
    }

    .section-card {
        padding: 1.8rem;
        margin-top: 1rem;
        background: #ffffff;
    }

    .section-eyebrow {
        color: var(--accent);
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.10em;
        font-size: 0.76rem;
        margin-bottom: 0.4rem;
    }

    .section-title {
        font-size: 2rem;
        font-weight: 900;
        margin-bottom: 0.55rem;
        color: #000000;
    }

    .section-description {
        color: var(--muted);
        font-size: 1.02rem;
        line-height: 1.65;
        margin-bottom: 0.5rem;
    }

    .result-card {
        padding: 1.6rem;
        margin-top: 0.45rem;
        background: #ffffff;
    }

    .result-title {
        font-size: 1.45rem;
        font-weight: 900;
        margin-bottom: 0.35rem;
        color: #000000;
    }

    .genre-pill {
        display: inline-block;
        padding: 0.36rem 0.7rem;
        margin: 0.18rem 0.22rem 0.18rem 0;
        border-radius: 999px;
        background: var(--accent-soft);
        border: 1px solid #bfdbfe;
        color: #1d4ed8;
        font-size: 0.84rem;
        font-weight: 800;
    }

    .big-score {
        font-size: 3rem;
        font-weight: 950;
        color: var(--green);
        margin: 0.45rem 0 0.2rem 0;
    }

    .stars {
        font-size: 2.1rem;
        letter-spacing: 0.08rem;
        color: var(--amber);
        margin-bottom: 0.75rem;
    }

    .soft-note {
        color: var(--muted);
        border: 1px solid var(--line);
        background: #f8fafc;
        border-radius: 16px;
        padding: 0.9rem 1rem;
        margin-top: 0.85rem;
    }

    .stButton > button {
        border-radius: 12px;
        padding: 0.7rem 1.15rem;
        font-weight: 800;
        border: 1px solid var(--line);
        background: #ffffff;
        color: #000000 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }

    .stButton > button:hover {
        border-color: var(--accent);
        background: var(--accent-soft);
        color: #000000 !important;
    }

    .stButton > button[kind="primary"] {
        background: #2563eb !important;
        border: 1px solid #2563eb !important;
        box-shadow: 0 8px 18px rgba(37, 99, 235, 0.20);
        color: white !important;
    }

    .stButton > button[kind="primary"] * {
        color: white !important;
    }

    [data-baseweb="select"] > div, [data-baseweb="input"] > div {
        border-radius: 12px;
        border-color: var(--line) !important;
        background-color: #ffffff !important;
        color: #000000 !important;
    }

    [data-baseweb="select"] *, [data-baseweb="input"] * {
        color: #000000 !important;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid var(--line);
        background: #ffffff;
    }

    [data-testid="stAlert"] {
        background: #f8fafc;
        color: #000000;
        border: 1px solid var(--line);
        border-radius: 14px;
    }

    [data-testid="stAlert"] * {
        color: #000000 !important;
    }

    /* Streamlit containers created with border=True */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--line) !important;
        border-radius: 18px !important;
        background: #ffffff !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.04);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def genre_pills(genres: str) -> str:
    items = [g.strip() for g in str(genres).replace("|", ",").split(",") if g.strip() and g.strip() != "(no genres listed)"]
    if not items:
        return '<span class="genre-pill">No genres listed</span>'
    return "".join(f'<span class="genre-pill">{g}</span>' for g in items[:6])


def rating_stars(score: float) -> str:
    full = int(score)
    half = 1 if score - full >= 0.5 else 0
    empty = max(0, 5 - full - half)
    return "★" * full + ("½" if half else "") + "☆" * empty


def profile_cards(ratings: pd.DataFrame):
    if ratings.empty:
        st.info("Add a few ratings first. Your predictions become more personal after the app learns your taste.")
        return

    r = ratings["rating"].astype(float)
    cards = [
        ("Movies rated", f"{len(r)}", "Keep rating movies to improve your profile."),
        ("Average rating", f"{r.mean():.2f}", "Your average score across saved ratings."),
        ("Rating range", f"{r.min():.1f} – {r.max():.1f}", "Minimum to maximum rating."),
    ]
    cols = st.columns(3)
    for col, (label, value, help_text) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-help">{help_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def section_intro(eyebrow: str, title: str, description: str):
    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-eyebrow">{eyebrow}</div>
            <div class="section-title">{title}</div>
            <div class="section-description">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------------
# Authentication section: still on Home, no separate page.
# -------------------------------------------------------------------
if not is_logged_in():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">🎬 Movies Recommender</div>
            <div class="hero-subtitle">
                Build your own movie profile, save personal ratings, and test three machine-learning tasks:
                clustering recommendations, like/dislike classification, and rating prediction with regression.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _login_form()
    st.stop()

user = current_user()
user_id = int(user["userId"])

# Sidebar navigation
with st.sidebar:
    st.markdown("### 🎬 Movies Recommender")
    st.caption(f"Logged in as **{user['username']}**")
    st.markdown("---")
    section = st.radio(
        "Choose an action",
        [
            "My Ratings",
            "Clustering Recommendation",
            "Classification",
            "Regression",
        ],
        label_visibility="visible",
    )
    st.markdown("---")
    st.caption("Use My Ratings first to make predictions more personal.")
    if st.button("Log out", key="logout_home", use_container_width=True):
        logout()
        st.rerun()

st.markdown(
    f"""
    <div class="topbar">
        <div>
            <div class="brand-title">🎬 Movies Recommender</div>
            <div class="brand-caption">Logged in as <strong>{user['username']}</strong></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

bundle, movie_catalog = load_movie_catalog()
ratings = get_user_ratings(user_id)

st.markdown(
    f"""
    <div class="hero-card">
        <div class="hero-title">Welcome, {user['username']} 👋</div>
        <div class="hero-subtitle">
            This dashboard uses your saved ratings together with the MovieLens dataset. Use clustering to receive recommendations,
            classification to check whether you would like one selected movie, and regression to estimate the rating you would give.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("## Your current profile")
profile_cards(ratings)


# -------------------------------------------------------------------
# My Ratings
# -------------------------------------------------------------------
if section == "My Ratings":
    section_intro(
        "Profile setup",
        "Add and manage your ratings",
        "Your ratings are saved to your account and used as the personal input for the machine-learning predictions.",
    )

    with st.container(border=True):
        st.subheader("Add or update a movie rating")
        st.caption("Choose a movie either by searching its title or by using the full dropdown list.")
        movie_id = movie_selectbox(movie_catalog, key="rating_movie")
        rating = st.slider("Your rating", min_value=0.5, max_value=5.0, value=4.0, step=0.5)
        submitted = st.button("Save rating", type="primary")

    if submitted:
        ok, msg = add_rating(user_id, movie_id, rating)
        st.success(msg) if ok else st.error(msg)
        st.rerun()

    st.subheader("Your saved ratings")
    ratings = get_user_ratings(user_id)

    if ratings.empty:
        st.info("No ratings saved yet.")
    else:
        display = ratings.merge(movie_catalog, on="movieId", how="left")
        display["title"] = display["title"].apply(clean_movie_title)
        display["timestamp"] = pd.to_datetime(display["timestamp"], unit="s", errors="coerce")
        display = display.sort_values("timestamp", ascending=False)
        st.dataframe(
            display[["title", "genres", "rating", "timestamp"]].rename(columns={
                "title": "Movie",
                "genres": "Genres",
                "rating": "Your rating",
                "timestamp": "Saved at",
            }),
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("Delete a saved rating"):
            delete_options = display[["movieId", "title"]].drop_duplicates("movieId").copy()
            delete_options["title"] = delete_options["title"].fillna("Unknown title").astype(str).apply(clean_movie_title)
            delete_options = delete_options.sort_values("title")

            delete_search = st.text_input(
                "Search saved movie title",
                key="delete_rating_search",
                placeholder="Type part of a saved movie title...",
            ).strip().lower()

            if delete_search:
                delete_options = delete_options[
                    delete_options["title"].str.lower().str.contains(delete_search, na=False, regex=False)
                ].copy()

            if delete_options.empty:
                st.warning("No saved ratings found for this search.")
            else:
                title_lookup = dict(zip(delete_options["movieId"].astype(int), delete_options["title"]))
                del_id = st.selectbox(
                    "Choose rating to delete",
                    delete_options["movieId"].astype(int).tolist(),
                    key="delete_rating_movie",
                    format_func=lambda movie_id: title_lookup.get(int(movie_id), "Unknown title"),
                )
                if st.button("Delete selected rating", type="secondary"):
                    ok, msg = delete_rating(user_id, int(del_id))
                    st.success(msg) if ok else st.error(msg)
                    st.rerun()

# -------------------------------------------------------------------
# Clustering
# -------------------------------------------------------------------
elif section == "Clustering Recommendation":
    section_intro(
        "Recommendation engine",
        "Clustering-based movie recommendations",
        "This is the only section that recommends movies. It groups users by similar rating behavior and suggests movies preferred by users in your cluster.",
    )

    clustering_model_name = bundle.get("best_clustering_recommender_name", "the exported best clustering model")
    st.info(f"Model used by the app: **{clustering_model_name}**")

    with st.container(border=True):
        top_n = st.slider("Number of recommendations", 5, 20, 10, key="cluster_topn")
        if st.button("Generate recommendations", type="primary"):
            try:
                ratings = get_user_ratings(user_id)
                recs = recommend_with_clustering(bundle, user_id, top_n=top_n, live_ratings_df=ratings)
                titles = recs["title"].dropna().astype(str).apply(clean_movie_title).reset_index(drop=True)
                if titles.empty:
                    st.info("No recommendations were found for your current profile yet.")
                else:
                    st.dataframe(
                        pd.DataFrame({"Recommended movies": titles}),
                        use_container_width=True,
                        hide_index=True,
                    )
            except Exception as exc:
                st.error(f"Recommendation failed: {type(exc).__name__}: {exc}")

# -------------------------------------------------------------------
# Classification
# -------------------------------------------------------------------
elif section == "Classification":
    section_intro(
        "Prediction task",
        "Classification: Will I like this movie?",
        "The classification model does not recommend a list of movies. It predicts whether you are likely to like one movie selected by you.",
    )

    left, right = st.columns([1.05, 1], gap="large")
    with left:
        with st.container(border=True):
            st.subheader("Predict one selected movie")
            movie_id = movie_selectbox(movie_catalog, key="class_movie")
            run_prediction = st.button("Predict like / not like", type="primary")

    with right:
        if run_prediction:
            try:
                ratings = get_user_ratings(user_id)
                info = predict_for_single_movie(bundle, user_id, movie_id, "Classification: predict liked/disliked", live_ratings_df=ratings)
                prob = float(info["prediction"])
                verdict = "Likely to like" if prob >= 0.5 else "May not like"
                status = "✅" if prob >= 0.5 else "⚠️"
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-title">{status} {verdict}</div>
                        <div style="color:#6b7280; margin-bottom:0.7rem;">{clean_movie_title(info.get('title', 'Selected movie'))}</div>
                        <div>{genre_pills(info.get('genres', 'N/A'))}</div>
                        <div class="big-score">{prob * 100:.1f}%</div>
                        <div class="soft-note">This is the estimated probability that you would like the selected movie based on your saved ratings.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if info.get("actual_rating") is not None:
                    st.caption(f"Your/dataset actual rating for this movie: {info['actual_rating']:.1f}")
            except Exception as exc:
                st.error(f"Prediction failed: {type(exc).__name__}: {exc}")
        else:
            st.info("Choose a movie and click the button to see the classification result here.")

# -------------------------------------------------------------------
# Regression
# -------------------------------------------------------------------
elif section == "Regression":
    section_intro(
        "Prediction task",
        "Regression: What rating would I give?",
        "The regression model does not recommend movies. It estimates the numerical rating you would probably give to one selected movie.",
    )

    left, right = st.columns([1.05, 1], gap="large")
    with left:
        with st.container(border=True):
            st.subheader("Predict one movie rating")
            movie_id = movie_selectbox(movie_catalog, key="reg_movie")
            run_prediction = st.button("Predict rating", type="primary")

    with right:
        if run_prediction:
            try:
                ratings = get_user_ratings(user_id)
                info = predict_for_single_movie(bundle, user_id, movie_id, "Regression: predict rating", live_ratings_df=ratings)
                pred = float(info["prediction"])
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-title">{clean_movie_title(info.get('title', 'Selected movie'))}</div>
                        <div style="color:#6b7280; margin-bottom:0.7rem;">Predicted rating</div>
                        <div>{genre_pills(info.get('genres', 'N/A'))}</div>
                        <div class="big-score">{pred:.2f} <span style="color:#1f2937;font-size:2rem;">/ 5.0</span></div>
                        <div class="stars">{rating_stars(pred)}</div>
                        <div class="soft-note">This is the rating you are likely to give based on your saved ratings and the selected movie features.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if info.get("actual_rating") is not None:
                    st.caption(f"Your/dataset actual rating for this movie: {info['actual_rating']:.1f}")
            except Exception as exc:
                st.error(f"Prediction failed: {type(exc).__name__}: {exc}")
        else:
            st.info("Choose a movie and click the button to see the predicted rating here.")
