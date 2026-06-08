import os
import zipfile
import requests
import pandas as pd

# MovieLens dataset URL
URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

# Folders
DATA_DIR = "data"
ZIP_PATH = os.path.join(DATA_DIR, "ml-latest-small.zip")
EXTRACT_DIR = os.path.join(DATA_DIR, "ml-latest-small")

os.makedirs(DATA_DIR, exist_ok=True)

# Download dataset
print("Downloading MovieLens dataset...")
response = requests.get(URL)

with open(ZIP_PATH, "wb") as file:
    file.write(response.content)

print("Download complete.")

# Extract zip
print("Extracting files...")
with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
    zip_ref.extractall(DATA_DIR)

print("Extraction complete.")

# Read CSV files
movies = pd.read_csv(os.path.join(EXTRACT_DIR, "movies.csv"))
ratings = pd.read_csv(os.path.join(EXTRACT_DIR, "ratings.csv"))
tags = pd.read_csv(os.path.join(EXTRACT_DIR, "tags.csv"))
links = pd.read_csv(os.path.join(EXTRACT_DIR, "links.csv"))

# ---- Extract year from title ----
movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
movies['title'] = movies['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True)

# Convert year to numeric (optional but recommended)
movies['year'] = movies['year'].astype('Int64')

# Merge datasets
movie_data = ratings.merge(movies, on="movieId", how="left")
movie_data = movie_data.merge(tags, on=["userId", "movieId"], how="left")
movie_data = movie_data.merge(links, on="movieId", how="left")

# Save final dataset
output_path = os.path.join(DATA_DIR, "movielens_full_data.csv")
movie_data.to_csv(output_path, index=False)

print(f"Final CSV saved to: {output_path}")
print(movie_data.head())