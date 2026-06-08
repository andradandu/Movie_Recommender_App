import numpy as np
import pandas as pd


def _as_dataframe(value):
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _feature_columns(bundle):
    return bundle.get("feature_cols") or bundle.get("feature_columns") or []


def _genre_columns(bundle):
    """Return the genre columns exported by the notebook."""
    return bundle.get("genre_cols") or [
        c for c in _feature_columns(bundle)
        if c not in {
            "userId", "movieId", "year", "rating", "liked",
            "movie_avg_rating", "movie_rating_count", "movie_rating_std",
            "user_avg_rating", "user_rating_count", "user_rating_std",
        }
    ]


def _compute_live_user_genre_profile(bundle, live_ratings_df):
    """
    Build the same type of user-level genre profile used in the notebook for clustering.

    In the notebook, clustering is trained on:
        data_model.groupby("userId")[genre_cols].mean()

    For a registered app user, their personal ratings table only contains movieId/rating,
    so we join those rated movies with movie_catalog and take the mean of the genre
    indicator columns. This lets the saved clustering model assign a real cluster to
    a new user who was not present in the original MovieLens data.
    """
    genre_cols = _genre_columns(bundle)
    movie_catalog = _as_dataframe(bundle.get("movie_catalog"))

    if not genre_cols:
        raise ValueError("genre_cols was not found in the exported bundle.")
    if live_ratings_df is None or live_ratings_df.empty or "movieId" not in live_ratings_df.columns:
        return pd.DataFrame([[0.0] * len(genre_cols)], columns=genre_cols)
    if movie_catalog.empty or "movieId" not in movie_catalog.columns:
        raise ValueError("movie_catalog was not found in the exported bundle.")

    rated_movies = live_ratings_df[["movieId"]].drop_duplicates().merge(
        movie_catalog[["movieId"] + [c for c in genre_cols if c in movie_catalog.columns]],
        on="movieId",
        how="left",
    )

    profile = rated_movies.reindex(columns=genre_cols).fillna(0).astype(float).mean(axis=0)
    return pd.DataFrame([profile], columns=genre_cols).fillna(0)


def _predict_user_cluster(bundle, user_id, live_ratings_df=None):
    """
    Return a cluster id for historical or newly registered users.

    Historical MovieLens users use user_cluster_table directly.
    New app users are transformed into a genre-preference vector, scaled with
    user_cluster_scaler, and assigned with best_clustering_recommender_model.predict(...)
    when the model supports prediction.
    """
    cluster_table = _as_dataframe(bundle.get("user_cluster_table"))

    if not cluster_table.empty and {"userId", "cluster"}.issubset(cluster_table.columns):
        historical_match = cluster_table.loc[cluster_table["userId"] == user_id, "cluster"]
        if not historical_match.empty:
            return historical_match.iloc[0]

    model = bundle.get("best_clustering_recommender_model")
    scaler = bundle.get("user_cluster_scaler")
    if model is None:
        raise ValueError("best_clustering_recommender_model was not found in the exported bundle.")
    if scaler is None:
        raise ValueError("user_cluster_scaler was not found in the exported bundle.")
    if not hasattr(model, "predict"):
        raise ValueError(
            "The selected clustering model cannot assign new users because it has no predict() method. "
            "Choose/export a clustering model such as K-Means or Gaussian Mixture."
        )

    user_profile = _compute_live_user_genre_profile(bundle, live_ratings_df)
    user_profile_scaled = scaler.transform(user_profile)
    return model.predict(user_profile_scaled)[0]


def _get_user_stats(bundle, user_id, live_ratings_df=None):
    """
    Compute user feature values for the model.

    Priority:
      1. If live_ratings_df is provided (logged-in user's own ratings), derive
         stats directly from those ratings — most accurate for new/custom users.
      2. Otherwise fall back to the pre-computed user_profile_stats in the bundle.
      3. If the user is not in the bundle either, use neutral defaults.
    """
    from utils.user_store import compute_user_stats_from_ratings

    feature_cols = _feature_columns(bundle)

    if live_ratings_df is not None and not live_ratings_df.empty:
        computed = compute_user_stats_from_ratings(live_ratings_df)
        row = {"userId": user_id, **computed}
    else:
        stats = _as_dataframe(bundle.get("user_profile_stats"))
        if not stats.empty and "userId" in stats.columns and user_id in set(stats["userId"]):
            row = stats.loc[stats["userId"] == user_id].iloc[0].to_dict()
        else:
            row = {
                "userId": user_id,
                "user_avg_rating": 3.5,
                "user_rating_count": 0,
                "user_rating_std": 0.0,
            }

    return {k: row.get(k, 0) for k in feature_cols if k.startswith("user_") or k == "userId"}


def build_candidate_frame(bundle, user_id, exclude_seen=True, live_ratings_df=None):
    """
    Build the feature matrix for all candidate movies for a given user.

    live_ratings_df — pass the logged-in user's personal ratings DataFrame so that
                      (a) user feature stats are computed from their own data, and
                      (b) movies they've already rated are excluded from recommendations.
    """
    movie_catalog = _as_dataframe(bundle.get("movie_catalog")).copy()
    if movie_catalog.empty:
        raise ValueError("movie_catalog was not found in the exported bundle.")

    feature_cols = _feature_columns(bundle)
    if not feature_cols:
        raise ValueError("feature_cols / feature_columns was not found in the exported bundle.")

    data_model = _as_dataframe(bundle.get("data_model"))

    if exclude_seen:
        seen = set()
        # Seen movies from historical bundle data
        if not data_model.empty and "userId" in data_model.columns and "movieId" in data_model.columns:
            seen |= set(data_model.loc[data_model["userId"] == user_id, "movieId"].unique())
        # Seen movies from the user's own live ratings
        if live_ratings_df is not None and not live_ratings_df.empty and "movieId" in live_ratings_df.columns:
            seen |= set(live_ratings_df["movieId"].unique())
        if seen:
            movie_catalog = movie_catalog[~movie_catalog["movieId"].isin(seen)].copy()

    user_values = _get_user_stats(bundle, user_id, live_ratings_df=live_ratings_df)
    for col, value in user_values.items():
        movie_catalog[col] = value

    X_candidates = movie_catalog.reindex(columns=feature_cols).fillna(0)
    return movie_catalog, X_candidates


def _scale_if_needed(bundle, X, technique):
    if technique == "regression" and bundle.get("best_regression_uses_scaled", False):
        return bundle["regression_export_scaler"].transform(X)
    if technique == "classification" and bundle.get("best_classification_uses_scaled", False):
        return bundle["classification_export_scaler"].transform(X)
    return X


def add_recommendation_explanation(result, method):
    result = result.copy()
    explanations = []

    for _, row in result.iterrows():
        title = row.get("title", "this movie")
        genres = row.get("genres", "unknown genres")
        parts = []

        if method == "regression":
            if "Predicted rating" in row:
                parts.append(f"predicted rating = {row['Predicted rating']:.2f}/5")
            if "movie_avg_rating" in row:
                parts.append(f"movie average rating = {row['movie_avg_rating']:.2f}")
            if "movie_rating_count" in row:
                parts.append(f"{int(row['movie_rating_count'])} historical ratings")
            parts.append(f"genres: {genres}")
            explanation = (
                f"{title} is recommended because the best regression model estimates a high rating for this user. "
                f"The score is based on user profile features, movie popularity/rating statistics, and genre indicators. "
                f"Main visible factors: {', '.join(parts)}."
            )

        elif method == "classification":
            if "Like probability" in row:
                parts.append(f"like probability = {row['Like probability']:.1%}")
            if "movie_avg_rating" in row:
                parts.append(f"movie average rating = {row['movie_avg_rating']:.2f}")
            if "movie_rating_count" in row:
                parts.append(f"{int(row['movie_rating_count'])} historical ratings")
            parts.append(f"genres: {genres}")
            explanation = (
                f"{title} is recommended because the best classification model estimates that the user will like it. "
                f"The decision is based on user behaviour, movie features, rating statistics, and genre indicators. "
                f"Main visible factors: {', '.join(parts)}."
            )

        else:
            if "cluster_average_rating" in row:
                parts.append(f"average rating in similar-user cluster = {row['cluster_average_rating']:.2f}")
            if "cluster_rating_count" in row:
                parts.append(f"{int(row['cluster_rating_count'])} ratings from similar users")
            if "movie_avg_rating" in row:
                parts.append(f"global movie average = {row['movie_avg_rating']:.2f}")
            parts.append(f"genres: {genres}")
            explanation = (
                f"{title} is recommended because users from the same preference cluster rated it highly. "
                f"The recommendation is based on clustering users with similar rating/genre behaviour, then ranking movies popular inside that cluster. "
                f"Main visible factors: {', '.join(parts)}."
            )

        explanations.append(explanation)

    result["Recommendation explanation"] = explanations
    return result


def recommend_with_regression(bundle, user_id, top_n=10, live_ratings_df=None):
    candidates, X = build_candidate_frame(bundle, user_id, live_ratings_df=live_ratings_df)
    model = bundle["best_regression_recommender_model"]
    X_model = _scale_if_needed(bundle, X, "regression")
    scores = model.predict(X_model)
    result = candidates.copy()
    result["Predicted rating"] = np.clip(scores, 0.5, 5.0)
    result = result.sort_values("Predicted rating", ascending=False).head(top_n)
    return add_recommendation_explanation(result, "regression")


def recommend_with_classification(bundle, user_id, top_n=10, live_ratings_df=None):
    candidates, X = build_candidate_frame(bundle, user_id, live_ratings_df=live_ratings_df)
    model = bundle["best_classification_recommender_model"]
    X_model = _scale_if_needed(bundle, X, "classification")

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_model)[:, 1]
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(X_model)
        scores = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    else:
        scores = model.predict(X_model)

    result = candidates.copy()
    result["Like probability"] = scores
    result = result.sort_values("Like probability", ascending=False).head(top_n)
    return add_recommendation_explanation(result, "classification")


def recommend_with_clustering(bundle, user_id, top_n=10, live_ratings_df=None):
    data_model = _as_dataframe(bundle.get("data_model"))
    cluster_table = _as_dataframe(bundle.get("user_cluster_table"))
    movie_catalog = _as_dataframe(bundle.get("movie_catalog")).copy()

    # Build seen set from both bundle and live ratings.
    # This keeps every user restricted to their own account history.
    seen = set()
    if not data_model.empty and {"userId", "movieId"}.issubset(data_model.columns):
        seen |= set(data_model.loc[data_model["userId"] == user_id, "movieId"].unique())
    if live_ratings_df is not None and not live_ratings_df.empty and "movieId" in live_ratings_df.columns:
        seen |= set(live_ratings_df["movieId"].unique())

    # IMPORTANT: this now works for both historical MovieLens users and newly
    # registered app users. New users are assigned to a cluster by computing
    # their genre profile and calling best_clustering_recommender_model.predict(...).
    cluster_id = _predict_user_cluster(bundle, user_id, live_ratings_df=live_ratings_df)

    similar_users = cluster_table.loc[cluster_table["cluster"] == cluster_id, "userId"]
    cluster_ratings = data_model[
        (data_model["userId"].isin(similar_users)) &
        (~data_model["movieId"].isin(seen))
    ]

    if not cluster_ratings.empty:
        scores = (
            cluster_ratings.groupby("movieId")
            .agg(cluster_average_rating=("rating", "mean"), cluster_rating_count=("rating", "count"))
            .reset_index()
        )
        scores["Cluster score"] = scores["cluster_average_rating"] * np.log1p(scores["cluster_rating_count"])
        scores["Predicted cluster"] = cluster_id
        result = scores.merge(movie_catalog, on="movieId", how="left")
        result = result.sort_values("Cluster score", ascending=False).head(top_n)
        return add_recommendation_explanation(result, "clustering")

    # Fallback only if the predicted cluster has no usable ratings.
    result = movie_catalog[~movie_catalog["movieId"].isin(seen)].copy() if seen else movie_catalog.copy()
    if "movie_avg_rating" in result.columns and "movie_rating_count" in result.columns:
        result["Cluster score"] = result["movie_avg_rating"] * np.log1p(result["movie_rating_count"])
        result["Predicted cluster"] = cluster_id
        result = result.sort_values("Cluster score", ascending=False).head(top_n)
    else:
        result = result.head(top_n)

    return add_recommendation_explanation(result, "clustering")


def get_actual_rating(bundle, user_id, movie_id, live_ratings_df=None):
    """Check live ratings first, then fall back to the bundle's data_model."""
    if live_ratings_df is not None and not live_ratings_df.empty:
        rows = live_ratings_df[live_ratings_df["movieId"] == movie_id]
        if not rows.empty:
            return float(rows.iloc[0]["rating"])

    data_model = _as_dataframe(bundle.get("data_model"))
    if data_model.empty or not {"userId", "movieId", "rating"}.issubset(data_model.columns):
        return None
    rows = data_model[(data_model["userId"] == user_id) & (data_model["movieId"] == movie_id)]
    if rows.empty:
        return None
    return float(rows.iloc[0]["rating"])


def predict_for_single_movie(bundle, user_id, movie_id, task, live_ratings_df=None):
    candidates, X = build_candidate_frame(
        bundle, user_id, exclude_seen=False, live_ratings_df=live_ratings_df
    )
    mask = candidates["movieId"] == movie_id
    if not mask.any():
        raise ValueError("Movie ID not found in the movie catalogue.")

    row = X.loc[mask]
    info = candidates.loc[mask].iloc[0].to_dict()
    info["actual_rating"] = get_actual_rating(bundle, user_id, movie_id, live_ratings_df=live_ratings_df)

    if task == "Regression: predict rating":
        model = bundle["best_regression_recommender_model"]
        row_model = _scale_if_needed(bundle, row, "regression")
        info["prediction"] = float(np.clip(model.predict(row_model)[0], 0.5, 5.0))
        info["prediction_description"] = (
            "The predicted rating is based on the selected user's profile statistics, the selected movie's rating/popularity "
            "statistics, and the genre columns used during model training."
        )
    elif task == "Classification: predict liked/disliked":
        model = bundle["best_classification_recommender_model"]
        row_model = _scale_if_needed(bundle, row, "classification")
        if hasattr(model, "predict_proba"):
            info["prediction"] = float(model.predict_proba(row_model)[:, 1][0])
        else:
            info["prediction"] = float(model.predict(row_model)[0])
        info["prediction_description"] = (
            "The like probability is based on the selected user's historical preferences, movie-level rating statistics, "
            "and genre indicators used by the classification model."
        )
    else:
        raise ValueError("Single-movie prediction is available for regression and classification only.")

    return info


def visible_feature_summary(bundle):
    feature_cols = _feature_columns(bundle)
    genre_cols = [c for c in feature_cols if c.startswith("genre_") or c in ["Action", "Comedy", "Drama", "Romance", "Sci-Fi"]]
    user_cols = [c for c in feature_cols if c.startswith("user_")]
    movie_cols = [c for c in feature_cols if c.startswith("movie_") or c in ["year"]]

    return {
        "User features": user_cols[:12],
        "Movie features": movie_cols[:12],
        "Genre features": genre_cols[:20],
    }
