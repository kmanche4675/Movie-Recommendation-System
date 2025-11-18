import pandas as pd
import os

os.chdir(r"C:\Users\kmanc\Movie Recommender\Movie-Recommendation-System")

movies = pd.read_csv("movies.csv")

movies = pd.read_csv(r"C:\Users\kmanc\Movie Recommender\Movie-Recommendation-System\movies.csv") # movieId, title, genres
tags = pd,read_csv('taga.csv')      # userId, movieId, tag, timestamp


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

indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def recommend(title, top_n=10):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies[['title', 'genres']].iloc[movie_indices]

recommend('The Matrix')
