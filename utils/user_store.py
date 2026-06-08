"""
user_store.py — persistent storage for registered users and their ratings.

Storage backend is chosen at runtime:
  - If DATABASE_URL is set in the environment → PostgreSQL (psycopg2)
  - Otherwise → CSV files in  data/users/
        data/users/users.csv        (userId, username, password_hash, created_at)
        data/users/user_ratings.csv (userId, movieId, rating, timestamp)

The CSV backend is the default; no configuration is required.
To switch to PostgreSQL set the DATABASE_URL env var before launching Streamlit:
    export DATABASE_URL="postgresql://user:pass@host:5432/dbname"
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
USERS_DIR = ROOT_DIR / "data" / "users"

# ── CSV helpers ──────────────────────────────────────────────────────────────

def _ensure_csv_dir():
    USERS_DIR.mkdir(parents=True, exist_ok=True)

def _users_path():
    return USERS_DIR / "users.csv"

def _ratings_path():
    return USERS_DIR / "user_ratings.csv"

def _read_users() -> pd.DataFrame:
    p = _users_path()
    if p.exists():
        return pd.read_csv(p, dtype={"userId": int})
    return pd.DataFrame(columns=["userId", "username", "password_hash", "created_at"])

def _write_users(df: pd.DataFrame):
    _ensure_csv_dir()
    df.to_csv(_users_path(), index=False)

def _read_ratings() -> pd.DataFrame:
    p = _ratings_path()
    if p.exists():
        return pd.read_csv(p, dtype={"userId": int, "movieId": int})
    return pd.DataFrame(columns=["userId", "movieId", "rating", "timestamp"])

def _write_ratings(df: pd.DataFrame):
    _ensure_csv_dir()
    df.to_csv(_ratings_path(), index=False)

# ── PostgreSQL helpers ───────────────────────────────────────────────────────

def _pg_conn():
    import psycopg2, psycopg2.extras
    return psycopg2.connect(os.environ["DATABASE_URL"])

def _ensure_pg_tables():
    conn = _pg_conn()
    with conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS app_users (
                    user_id      SERIAL PRIMARY KEY,
                    username     TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at   BIGINT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS app_ratings (
                    id        SERIAL PRIMARY KEY,
                    user_id   INT NOT NULL REFERENCES app_users(user_id),
                    movie_id  INT NOT NULL,
                    rating    REAL NOT NULL,
                    ts        BIGINT NOT NULL,
                    UNIQUE (user_id, movie_id)
                );
            """)
    conn.close()

# ── Public interface ─────────────────────────────────────────────────────────

def _use_pg() -> bool:
    return bool(os.environ.get("DATABASE_URL"))

def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# ─── Users ───────────────────────────────────────────────────────────────────

def register_user(username: str, password: str) -> tuple[bool, str]:
    """Register a new user. Returns (success, message)."""
    username = username.strip()
    if not username or not password:
        return False, "Username and password cannot be empty."

    if _use_pg():
        _ensure_pg_tables()
        import psycopg2
        conn = _pg_conn()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO app_users (username, password_hash, created_at) VALUES (%s,%s,%s)",
                        (username, _hash(password), int(time.time()))
                    )
            return True, "Account created successfully."
        except psycopg2.errors.UniqueViolation:
            return False, "Username already taken."
        finally:
            conn.close()
    else:
        users = _read_users()
        if username in users["username"].values:
            return False, "Username already taken."
        new_id = int(users["userId"].max() + 1) if not users.empty else 100001
        new_row = pd.DataFrame([{
            "userId": new_id,
            "username": username,
            "password_hash": _hash(password),
            "created_at": int(time.time()),
        }])
        _write_users(pd.concat([users, new_row], ignore_index=True))
        return True, "Account created successfully."

def login_user(username: str, password: str) -> tuple[bool, dict | None]:
    """Verify credentials. Returns (success, user_dict | None)."""
    username = username.strip()
    if _use_pg():
        _ensure_pg_tables()
        conn = _pg_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, username FROM app_users WHERE username=%s AND password_hash=%s",
                (username, _hash(password))
            )
            row = cur.fetchone()
        conn.close()
        if row:
            return True, {"userId": row[0], "username": row[1]}
        return False, None
    else:
        users = _read_users()
        match = users[(users["username"] == username) & (users["password_hash"] == _hash(password))]
        if match.empty:
            return False, None
        row = match.iloc[0]
        return True, {"userId": int(row["userId"]), "username": row["username"]}

def get_all_registered_users() -> pd.DataFrame:
    if _use_pg():
        _ensure_pg_tables()
        conn = _pg_conn()
        df = pd.read_sql("SELECT user_id AS userId, username FROM app_users", conn)
        conn.close()
        return df
    return _read_users()[["userId", "username"]]

# ─── Ratings ─────────────────────────────────────────────────────────────────

def add_rating(user_id: int, movie_id: int, rating: float) -> tuple[bool, str]:
    """Add or update a rating. Returns (success, message)."""
    if not (0.5 <= rating <= 5.0):
        return False, "Rating must be between 0.5 and 5.0."
    ts = int(time.time())

    if _use_pg():
        _ensure_pg_tables()
        conn = _pg_conn()
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO app_ratings (user_id, movie_id, rating, ts)
                    VALUES (%s,%s,%s,%s)
                    ON CONFLICT (user_id, movie_id) DO UPDATE SET rating=EXCLUDED.rating, ts=EXCLUDED.ts
                """, (user_id, movie_id, rating, ts))
        conn.close()
        return True, "Rating saved."
    else:
        ratings = _read_ratings()
        mask = (ratings["userId"] == user_id) & (ratings["movieId"] == movie_id)
        if mask.any():
            ratings.loc[mask, "rating"] = rating
            ratings.loc[mask, "timestamp"] = ts
        else:
            new_row = pd.DataFrame([{
                "userId": user_id, "movieId": movie_id,
                "rating": rating, "timestamp": ts,
            }])
            ratings = pd.concat([ratings, new_row], ignore_index=True)
        _write_ratings(ratings)
        return True, "Rating saved."

def delete_rating(user_id: int, movie_id: int) -> tuple[bool, str]:
    if _use_pg():
        _ensure_pg_tables()
        conn = _pg_conn()
        with conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM app_ratings WHERE user_id=%s AND movie_id=%s", (user_id, movie_id))
        conn.close()
        return True, "Rating deleted."
    else:
        ratings = _read_ratings()
        before = len(ratings)
        ratings = ratings[~((ratings["userId"] == user_id) & (ratings["movieId"] == movie_id))]
        if len(ratings) == before:
            return False, "Rating not found."
        _write_ratings(ratings)
        return True, "Rating deleted."

def get_user_ratings(user_id: int) -> pd.DataFrame:
    if _use_pg():
        _ensure_pg_tables()
        conn = _pg_conn()
        df = pd.read_sql(
            "SELECT movie_id AS movieId, rating, ts AS timestamp FROM app_ratings WHERE user_id=%s",
            conn, params=(user_id,)
        )
        conn.close()
        return df
    ratings = _read_ratings()
    return ratings[ratings["userId"] == user_id][["movieId", "rating", "timestamp"]].copy()

# ─── Feature computation for ML ──────────────────────────────────────────────

def compute_user_stats_from_ratings(ratings_df: pd.DataFrame) -> dict:
    """
    Given a user's ratings DataFrame (columns: rating), compute the same
    user_avg_rating / user_rating_count / user_rating_std features the
    notebook produced for historical users.
    """
    if ratings_df.empty:
        return {"user_avg_rating": 3.5, "user_rating_count": 0, "user_rating_std": 0.0}
    r = ratings_df["rating"].dropna().astype(float)
    return {
        "user_avg_rating": float(r.mean()),
        "user_rating_count": int(len(r)),
        "user_rating_std": float(r.std(ddof=0)) if len(r) > 1 else 0.0,
    }
