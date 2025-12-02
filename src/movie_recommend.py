import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "ml-latest-small")

movies_path = os.path.join(DATA_DIR, "movies.csv")
tags_path = os.path.join(DATA_DIR, "tags.csv")

movies = pd.read_csv(movies_path) # movieId, title, genres
tags = pd.read_csv(tags_path)      # userId, movieId, tag, timestamp

# Aggregate tags per movie
tag_df = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()

# Merge tags with movies
movies = movies.merge(tag_df, on='movieId', how='left')
movies['tag'] = movies['tag'].fillna('')  # Fill NaN tags with empty string

# Combine genres and tags into one text column
movies['features'] = (movies['genres'].fillna('') + ' ' + movies['tag'].fillna('')).str.strip()

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['features'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

title_keys = movies['title'].astype(str).str.strip().str.lower()
indices = pd.Series(movies.index.values, index=title_keys).drop_duplicates()

def recommend(title, top_n=10):
    title_key = str(title).strip().lower()
    if title_key not in indices:
        raise ValueError(f"Title not found: {title}")
    idx = indices[title_key]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies[['title', 'genres']].iloc[movie_indices]

recommend('The Matrix')
