import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "ml-latest-small")

movies_path = os.path.join(DATA_DIR, "movies.csv")
tags_path = os.path.join(DATA_DIR, "tags.csv")

movies = pd.read_csv(movies_path) # movieId, title, genres
tags = pd.read_csv(tags_path)  # userId, movieId, tag, timestamp

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
        # fallback: case-insensitive substring match
        titles_lower = movies['title'].astype(str).str.lower()
        mask = titles_lower.str.contains(title_key, na=False)
        if not mask.any():
            raise ValueError(f"Title not found: {title}")
        canonical = movies['title'][mask].iloc[0].strip().lower()
        idx = indices[canonical]
    else:
        idx = indices[title_key]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies[['title', 'genres']].iloc[movie_indices]

# Collaborative Filtering
ratings_path = os.path.join(DATA_DIR, "ratings.csv")
ratings = pd.read_csv(ratings_path)  

# User-item matrix
user_item = ratings.pivot_table(index="userId", columns="movieId", values="rating")
# Normalize per user (center by mean)
user_item_centered = user_item.sub(user_item.mean(axis=1), axis=0).fillna(0)

# Compute item-item cosine similarity
item_vectors = user_item_centered.values.T  # items x users
collab_sim = cosine_similarity(item_vectors, item_vectors)
movie_ids = user_item_centered.columns.to_numpy()
movieId_to_collab_idx = {int(mid): i for i, mid in enumerate(movie_ids)}

def recommend_by_title(title, alpha=0.6, top_n=10):
    """Blend content and collaborative item-item scores for a seed title."""
    title_key = str(title).strip().lower()
    if title_key not in indices:
        raise ValueError(f"Title not found: {title}")
    content_idx = int(indices[title_key])
    seed_movie_id = int(movies.iloc[content_idx]["movieId"])

    # Content scores
    content_scores = cosine_sim[content_idx]

    # Collaborative scores aligned to movies order
    if seed_movie_id in movieId_to_collab_idx:
        collab_idx = movieId_to_collab_idx[seed_movie_id]
        collab_scores_items = collab_sim[collab_idx]
        collab_scores = pd.Series(0.0, index=movies.index)
        # Build map movieId -> row index for speed
        id_to_row = {int(r.movieId): i for i, r in movies[["movieId"]].reset_index(drop=True).itertuples(index=False)}
        for i in range(len(movie_ids)):
            mid = int(movie_ids[i])
            row_idx = id_to_row.get(mid)
            if row_idx is not None:
                collab_scores.iloc[row_idx] = float(collab_scores_items[i])
        collab_scores = collab_scores.to_numpy()
    else:
        collab_scores = 0.0 * content_scores

    hybrid_scores = alpha * content_scores + (1.0 - alpha) * collab_scores

    candidates = pd.DataFrame({
        "movieId": movies["movieId"],
        "title": movies["title"],
        "genres": movies["genres"],
        "score": hybrid_scores,
    })
    candidates = candidates[candidates["movieId"] != seed_movie_id]
    candidates = candidates.sort_values("score", ascending=False).head(top_n)
    return candidates.reset_index(drop=True)

def recommend_by_likes(liked_titles, alpha=0.6, top_n=10):
    """Hybrid recommendations from multiple liked titles (content + collaborative)."""
    if not liked_titles:
        raise ValueError("Provide at least one liked title.")
    liked_content_idxs = []
    liked_collab_idxs = []
    liked_movie_ids = []
    for t in liked_titles:
        key = str(t).strip().lower()
        if key in indices:
            ci = int(indices[key])
            liked_content_idxs.append(ci)
            mid = int(movies.iloc[ci]["movieId"])
            liked_movie_ids.append(mid)
            if mid in movieId_to_collab_idx:
                liked_collab_idxs.append(movieId_to_collab_idx[mid])

    if not liked_content_idxs:
        raise ValueError("None of the liked titles were found.")

    # Content user profile
    user_vec = tfidf_matrix[liked_content_idxs].mean(axis=0)
    content_user_scores = cosine_similarity(user_vec, tfidf_matrix).ravel()

    # Collaborative user profile
    if liked_collab_idxs:
        collab_user_scores_items = collab_sim[liked_collab_idxs].mean(axis=0)
        collab_scores = pd.Series(0.0, index=movies.index)
        id_to_row = {int(r.movieId): i for i, r in movies[["movieId"]].reset_index(drop=True).itertuples(index=False)}
        for i in range(len(movie_ids)):
            mid = int(movie_ids[i])
            row_idx = id_to_row.get(mid)
            if row_idx is not None:
                collab_scores.iloc[row_idx] = float(collab_user_scores_items[i])
        collab_user_scores = collab_scores.to_numpy()
    else:
        collab_user_scores = 0.0 * content_user_scores

    hybrid_scores = alpha * content_user_scores + (1.0 - alpha) * collab_user_scores

    # Exclude liked items
    exclude_ids = set(liked_movie_ids)
    candidates = pd.DataFrame({
        "movieId": movies["movieId"],
        "title": movies["title"],
        "genres": movies["genres"],
        "score": hybrid_scores,
    })
    candidates = candidates[~candidates["movieId"].isin(exclude_ids)]
    candidates = candidates.sort_values("score", ascending=False).head(top_n)
    return candidates.reset_index(drop=True)

if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if args and args[0] == "--likes":
        likes = args[1:]
        df = recommend_by_likes(likes, alpha=0.6, top_n=10)
        print(df.to_string(index=False))
    elif args and args[0] == "--hybrid":
        title = " ".join(args[1:]) if len(args) > 1 else "The Matrix"
        df = recommend_by_title(title, alpha=0.6, top_n=10)
        print(df.to_string(index=False))
    else:
        title = " ".join(args) if args else "The Matrix"
        df = recommend(title, top_n=10)
        print(df.to_string(index=False))
