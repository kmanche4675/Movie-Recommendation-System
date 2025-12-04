import pandas as pd
import os
import difflib

# Set working directory to project root
os.chdir(r"C:\Users\kmanc\Movie Recommender\Movie-Recommendation-System")

# Load movies.csv from data/ml-latest-small
movies = pd.read_csv("data/ml-latest-small/movies.csv")   # movieId, title, genres

# Load tags.csv from data/ml-latest-small
tags = pd.read_csv("data/ml-latest-small/tags.csv")       # userId, movieId, tag, timestamp

# Aggregate tags per movie
tag_df = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()

# Merge tags with movies
movies = movies.merge(tag_df, on='movieId', how='left')
movies['tag'] = movies['tag'].fillna('')  # Fill NaN tags with empty string

# Combine genres and tags into one text column
movies['features'] = movies['genres'] + ' ' + movies['tag']

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['features'])

from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create a clean_title column without year formatting
movies['clean_title'] = movies['title'].str.replace(r"\(\d{4}\)", "", regex=True).str.strip()

# Normalize titles to lowercase for case-insensitive matching
indices = pd.Series(movies.index, index=movies['clean_title'].str.lower()).drop_duplicates()

def recommend(title, top_n=10):
    # Normalize input
    title = title.strip().lower()
    if title not in indices:
        # Suggest close matches if not found
        close_matches = difflib.get_close_matches(title, indices.index, n=3, cutoff=0.6)
        if close_matches:
            return f"Movie '{title}' not found. Did you mean: {', '.join(close_matches)}?"
        else:
            return f"Movie '{title}' not found in dataset."
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies[['title', 'genres']].iloc[movie_indices]

# Example usage
print(recommend("The Matrix"))

if __name__ == "__main__":
    user_title = input("Enter a movie title: ")
    n = input("How many recommendations would you like? (default 10): ")
    n = int(n) if n.isdigit() else 10
    print(recommend(user_title, top_n=n))
