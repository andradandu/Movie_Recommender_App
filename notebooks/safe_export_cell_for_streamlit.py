# Paste this as the final cell of your notebook if the app needs a fresh export.
# It keeps the model objects in joblib but exports result tables in a safer format.

import os
import joblib

os.makedirs("models", exist_ok=True)

def safe_records(df):
    try:
        return df.copy().astype(str).to_dict("records")
    except Exception:
        return df

bundle = {
    # Best recommender models from Option B
    "best_regression_recommender_model": best_regression_recommender_model,
    "best_classification_recommender_model": best_classification_recommender_model,
    "best_clustering_recommender_model": best_clustering_recommender_model,

    "best_regression_recommender_name": best_regression_recommender_name,
    "best_classification_recommender_name": best_classification_recommender_name,
    "best_clustering_recommender_name": best_clustering_recommender_name,

    "best_regression_uses_scaled": best_regression_uses_scaled,
    "best_classification_uses_scaled": best_classification_uses_scaled,

    "regression_export_scaler": regression_export_scaler,
    "classification_export_scaler": classification_export_scaler,

    "feature_cols": list(feature_cols),

    "movie_catalog": movie_catalog,
    "user_profile_stats": user_profile_stats,
    "user_cluster_table": user_cluster_table,
    "data_model": data_model,

    # Tables shown in Streamlit
    "all_recommender_results_df": all_recommender_results_df,
    "best_option_b_by_technique": best_option_b_by_technique,
    "regression_results_df": regression_results_df,
    "classification_results_df": classification_results_df,
    "clustering_results_df": clustering_results_df,
}

joblib.dump(bundle, "models/movie_recommender_bundle.joblib")
print("✅ Exported models/movie_recommender_bundle.joblib")
