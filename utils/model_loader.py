from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "movie_recommender_bundle.joblib"
NOTEBOOK_PATH = ROOT_DIR / "notebooks" / "data_mining_ml_clean_option_b_only.ipynb"
DATA_PATH = ROOT_DIR / "data" / "movielens_full_data.csv"

@st.cache_resource(show_spinner=False)
def load_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data(show_spinner=False)
def load_dataset():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    bundle = load_bundle()
    if bundle is not None:
        data_model = bundle.get("data_model")
        if isinstance(data_model, pd.DataFrame):
            return data_model.copy()
        movie_catalog = bundle.get("movie_catalog")
        if isinstance(movie_catalog, pd.DataFrame):
            return movie_catalog.copy()
    return pd.DataFrame()

def require_bundle():
    bundle = load_bundle()
    if bundle is None:
        st.error("Model bundle not found: models/movie_recommender_bundle.joblib")
        st.info(
            "Open notebooks/data_mining_ml_clean_option_b_only.ipynb, run all cells, "
            "and make sure the final export cell creates models/movie_recommender_bundle.joblib."
        )
        st.stop()
    return bundle

def as_dataframe(value):
    """Convert exported tables safely, whether the notebook saved DataFrames or list-of-dicts."""
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, list):
        return pd.DataFrame(value)
    if isinstance(value, dict):
        return pd.DataFrame(value)
    return pd.DataFrame()
