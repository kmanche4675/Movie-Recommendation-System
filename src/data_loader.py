import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np

# --- STANDARDIZED PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "processed")
MERGED_DATA_FILE = os.path.join(PROCESSED_DATA_PATH, 'merged_data_clean.csv')

# --- MODEL ARTIFACT PATHS (Saves the model for the Hybrid Lead) ---
CBF_ARTIFACTS_PATH = os.path.join(SCRIPT_DIR, "..", "data", "cbf_artifacts")
COSINE_SIM_FILE = os.path.join(CBF_ARTIFACTS_PATH, 'cbf_cosine_sim.pkl')
MOVIE_INDEX_FILE = os.path.join(CBF_ARTIFACTS_PATH, 'cbf_movie_indices.pkl')
MOVIE_DF_FILE = os.path.join(CBF_ARTIFACTS_PATH, 'cbf_unique_movies.pkl')


def get_unique_movie_data():
    """Loads and prepares data required for pure Content-Based Filtering."""
    try:
        # CRITICAL: Loads the clean, merged file created by the data_loader
        df = pd.read_csv(MERGED_DATA_FILE)
        
        # Filter for unique movies (dropping duplicate rating rows)
        movies = df[['movieId', 'title', 'genres', 'tag']].drop_duplicates(subset='movieId').reset_index(drop=True)
        
        # --- FEATURE ENGINEERING ---
        movies['genres'] = movies['genres'].fillna('')
        movies['tag'] = movies['tag'].fillna('')
        # Combine genres and tags into one text column for feature extraction
        movies['features'] = movies['genres'] + ' ' + movies['tag']
        
        print(f"[CBF Lead] Data Prepared: {len(movies)} unique movies loaded.")
        return movies
    except FileNotFoundError:
        print(f"ERROR: Processed data not found at {MERGED_DATA_FILE}.")
        print("ACTION: Please ensure you run 'python src/data_loader.py' successfully first.")
        return None

def build_and_save_cbf_model(movies):
    """
    Builds the CBF model using TF-IDF and Cosine Similarity, and saves artifacts.
    """
    print("\n[CBF Lead] Building Content-Based Model...")

    # --- 1. TF-IDF Vectorization ---
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['features'])
    
    # --- 2. Cosine Similarity Calculation ---
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Create title index map
    title_keys = movies['title'].astype(str).str.strip().str.lower()
    indices = pd.Series(movies.index, index=title_keys).drop_duplicates()
    
    # --- 3. Save Artifacts for Teammate ---
    if not os.path.exists(CBF_ARTIFACTS_PATH):
        os.makedirs(CBF_ARTIFACTS_PATH)
        
    with open(COSINE_SIM_FILE, 'wb') as f:
        pickle.dump(cosine_sim, f)
    with open(MOVIE_INDEX_FILE, 'wb') as f:
        pickle.dump(indices, f)
    with open(MOVIE_DF_FILE, 'wb') as f:
        pickle.dump(movies, f)
        
    print(f"[CBF Lead] Model artifacts saved.")
    return cosine_sim, indices, movies

def load_cbf_artifacts():
    """Helper to load the saved CBF model components."""
    try:
        with open(COSINE_SIM_FILE, 'rb') as f:
            cosine_sim = pickle.load(f)
        with open(MOVIE_INDEX_FILE, 'rb') as f:
            indices = pickle.load(f)
        with open(MOVIE_DF_FILE, 'rb') as f:
            movies_df = pickle.load(f)
        return cosine_sim, indices, movies_df
    except FileNotFoundError:
        return None, None, None

def get_cbf_recommendations(seed_movie_ids: list[int], top_n: int = 10):
    """
    Public function for the Hybrid Lead to call. Accepts a list of MovieIDs 
    and returns recommendations based on the average profile of those seeds.
    """
    cosine_sim_matrix, indices_series, movies_df = load_cbf_artifacts()
    if cosine_sim_matrix is None or movies_df is None: return []
    
    # Map MovieIDs to the internal matrix indices
    movie_id_to_index = {mid: idx for idx, mid in movies_df['movieId'].items()}
    cbf_indices = [movie_id_to_index.get(mid) for mid in seed_movie_ids if mid in movie_id_to_index]
    if not cbf_indices: return []

    # Calculate the average score vector from the seeds
    # This creates the "User Profile" vector by averaging the vectors of liked movies
    avg_sim_score = cosine_sim_matrix[cbf_indices].mean(axis=0)

    # Get the scores and remove seeds
    sim_scores = list(enumerate(avg_sim_score))
    sim_scores = [s for s in sim_scores if movies_df.iloc[s[0]]['movieId'] not in seed_movie_ids]
    sim_scores.sort(key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[:top_n]
    
    # Format output
    recs_df = pd.DataFrame([
        {'movieId': movies_df.iloc[i[0]]['movieId'], 
         'title': movies_df.iloc[i[0]]['title'], 
         'score': i[1]} 
        for i in sim_scores
    ])
    
    return [
        {"movieId": int(row.movieId), "title": row.title, "score": float(row.score)}
        for _, row in recs_df.iterrows()
    ]


if __name__ == "__main__":
    # --- EXECUTION BLOCK: Run this to train and save the model ---
    movies_df_unique = get_unique_movie_data()
    
    if movies_df_unique is not None:
        # STEP 2: Build Model and Save Artifacts
        cosine_sim_matrix, indices_series, movies_df_saved = build_and_save_cbf_model(movies_df_unique)

        # Self-Test
        TARGET_IDS = [1, 3114] # Example Movie IDs (e.g., Toy Story and Toy Story 3)
        
        recs = get_cbf_recommendations(TARGET_IDS, top_n=5)
        
        print(f"\n--- Content-Based Self-Test for Seeds {TARGET_IDS} ---")
        if recs:
            print(pd.DataFrame(recs))
        else:
            print("Test failed: No recommendations returned.")