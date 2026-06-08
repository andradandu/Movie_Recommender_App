# Movie Recommender Streamlit App

Updated page flow:

1. **Authentication** (`Home.py`) — login/register only.
2. **Home** — explains the app and the three Data Mining methods.
3. **My Ratings** — the logged-in user adds, updates, views, and deletes their own ratings.
4. **Clustering Recommendation** — recommends movies using similar-user cluster behaviour.
5. **Classification** — predicts whether the logged-in user will like a movie.
6. **Regression** — predicts the rating the logged-in user would give a movie.

The user cannot select another user for predictions. All predictions use the current logged-in account ID and that user's saved ratings.

## Run

```bash
pip install -r requirements.txt
streamlit run Home.py
```

By default, users and ratings are saved in CSV files under `data/users/`. To use PostgreSQL, set `DATABASE_URL` before running Streamlit.


## Single-page version
The app now uses only `Home.py`. Authentication, rating management, clustering, classification, and regression are all accessible inside Home.
