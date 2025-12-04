import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "processed")
MOVIES_FILE = os.path.join(PROCESSED_DATA_PATH, "movies_clean.csv")

# Load movies
movies_df = pd.read_csv(MOVIES_FILE)

# Build TF-IDF matrix on genres
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies_df["genres"].fillna(""))

# Compute cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def get_content_recommendations(title, movies_df, cosine_sim, n=5):
    idx = movies_df[movies_df["title"].str.contains(title, case=False, na=False)].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx]
    sim_indices = [i[0] for i in sim_scores[1:n+1]]
    return movies_df.iloc[sim_indices][["title", "genres"]]

if __name__ == "__main__":
    recs = get_content_recommendations("Avengers: Infinity War", movies_df, cosine_sim, n=5)
    print("\n--- Content-Based Recommendations for 'Avengers: Infinity War' ---")
    print(recs)
