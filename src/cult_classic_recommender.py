#  cult_classic_recommender.py

import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from collections import defaultdict
import json, gzip, math, os
from pathlib import Path
from data_loader import load_movielens_data

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
TMDB_CACHE_FILE = DATA_DIR / "tmdb_metadata.csv"                 # CSV generated from tuner
CLASSIC_MODEL_PATH = MODELS_DIR / "classic_svd_best.pkl"     
CULT_MODEL_PATH = MODELS_DIR / "cult_svd_best.pkl"
CULT_CONFIG_PATH = MODELS_DIR / "cult_classic_best.json"

TOP_N        = 20                               # How many recs to return
POPULARITY_CAP = 1000                           # movies with >CAP votes are “mainstream”
OBSCURITY_EXP  = 0.6            # higher -> stronger obscurity push. key for cult classic
MIN_VOTES      = 50                             # TMDB quality filter
DEFAULT_PARAMS={
    "n_factors": 120,
    "n_epochs": 30,
    "lr_all": .005,
    "reg_all": .1,
    "obscurity_exp": .6,
    "min_votes": 50,
}

HAS_TMDB = False
    
def load_best_params(path:Path):
    """Load tuned paramerters; fall back to defaults if missing."""
    if path.is_file():
        print(f"Loading best params from {path}...")
        with open(path, "r") as f:
            raw = json.load(f)
        try:
            svd_cfg = raw["svd"]
            cult_cfg = raw["cult"]
            params = {
                "n_factors": svd_cfg["n_factors"],
                "n_epochs": svd_cfg["n_epochs"],
                "lr_all": svd_cfg["lr_all"],
                "reg_all": svd_cfg["reg_all"],
                "obscurity_exp": cult_cfg["obscurity_exp"],
                "min_votes": cult_cfg["min_votes"],
            }
            return params
            
        except (KeyError, TypeError):
            print(f"Best params file corrupted. Run cult_recommender_tuner.py to re-generate. Currently Loading default params...")
    else:
        print(f"Best params file missing. Run cult_recommender_tuner.py to generate. Currently Loading default params...")
        
    return DEFAULT_PARAMS.copy()
        
best_params = load_best_params(CULT_CONFIG_PATH)

N_FACTORS =best_params["n_factors"]
N_EPOCHS = best_params["n_epochs"]
LR_ALL = best_params["lr_all"]
REG_ALL = best_params["reg_all"]
OBSCURITY_EXP = best_params["obscurity_exp"]
MIN_VOTES = best_params["min_votes"]

print(f"Using SVD params : n_factors = {N_FACTORS}, n_epochs = {N_EPOCHS}, lr_all = {LR_ALL}, reg_all = {REG_ALL}")

print(f"Using cult params: obscurity_exp = {OBSCURITY_EXP}, min_votes = {MIN_VOTES}")

# 1. Load & align data

print("Loading MovieLens …")

data = load_movielens_data(DATA_DIR / "ml-latest-small")

ratings = data['ratings']
movies = data['movies']
links = data['links']
tags = data.get('tags')

# Cleanup on links
links = links.dropna(subset=['tmdbId'])
links['tmdbId']= links['tmdbId'].astype(int)

print(f"Loaded {len(ratings):,} ratings, {len(movies):,} movies, {len(links):,} links")

if os.path.exists(TMDB_CACHE_FILE):
    print("Loading cached TMDB metadata …")
    tmdb_df = pd.read_csv(TMDB_CACHE_FILE)
    HAS_TMDB = True

else:
    print(f"TMDB metadata cahce not found: {TMDB_CACHE_FILE}\n"
          "Proceeding WITHOUT cult-classic weighting (ratings only recommender).")
    tmdb_df = None
    HAS_TMDB = False

# Merge TMDB stats onto MovieLens items

if HAS_TMDB:
    item_stats = links.merge(tmdb_df, on='tmdbId', how='left')
    item_stats = item_stats.merge(movies[['movieId', 'title']], on='movieId')

    for column in ["vote_average", "vote_count", "popularity"]:
        item_stats[column] = pd.to_numeric(item_stats[column], errors="coerce").fillna(0.0)
    if "poster_path" not in item_stats.columns:
        item_stats["poster_path"] = None
    else:
        item_stats["poster_path"] = item_stats["poster_path"].fillna(None)
else:
    item_stats = links.merge(movies[["movieId", "title"]], on="movieId")
    item_stats["vote_average"] = 0.0
    item_stats["vote_count"] = 0
    item_stats["popularity"] = 0.0

# 2. Train SVD (Singular Value Decomposition) on MovieLens

reader = Reader(rating_scale=(0.5, 5.0))
data   = Dataset.load_from_df(ratings[['userId','movieId','rating']], reader)

trainset, testset = train_test_split(data, test_size=.2, random_state=42)

def train_SVD_and_eval(trainset, testset):
    
    print("\nTraining SVD…")
    algo = SVD(
        n_factors = N_FACTORS,
        n_epochs = N_EPOCHS,
        lr_all = LR_ALL,
        reg_all = REG_ALL,
        random_state = 42
    )
    
    algo.fit(trainset)
    
    # 2.1 Evaluating the training
    
    print("\nEvaluating...")
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose= False)     # Root Mean Square Error
    mae= accuracy.mae(predictions, verbose=False)         # Mean Absolute Error
    print(f"Test RMSE: {rmse:.4f} | Test MAE: {mae:.4f}")

    return algo, rmse, mae
    
# 3. Helper: obscurity / quality weights

def cult_score(row):
    
    # quality (TMDB average, weighted by votes)
    if row['vote_count'] < MIN_VOTES:
        return 0.0
    quality = row['vote_average'] / 10.0   # 0–1
    # obscurity (inverse log-popularity)
    pop_raw = row["popularity"]
    pop = max(min(pop_raw, POPULARITY_CAP), 1.0)
    obscurity = 1.0 / (math.log1p(pop) ** OBSCURITY_EXP)

    return quality * obscurity

if HAS_TMDB:
    
    print("Computing cult weights from TMDB metadata...")
    total_with_tmdb = len(item_stats)
    # Pre-compute for every movie
    item_stats['cult_weight'] = item_stats.apply(cult_score, axis=1)
    item_stats = item_stats[item_stats['cult_weight'] > 0].copy()
    remaining_total = len(item_stats)

    print(
        f"Movies with TMDB stats: {total_with_tmdb:,} | "
        f"Remaining movies with cult_weight>0: {remaining_total:,}"
    )

    top_preview=(
        item_stats.sort_values("cult_weight", ascending=False)
        [["title", "vote_average","vote_count","popularity","cult_weight"]]
        .head(5)
    )
    print("\nTop5 cult_weighted movies:")
    print(top_preview.to_string(index=False, float_format="%.4f"))

else:
    print("No TMDB metadata available. Setting cult-weight = 1.0 for all movies.")
    item_stats["cult_weight"] = 1.0

# 4. Recommendation function

def recommend_cult_classics(algo, trainset, user_id, top_n=TOP_N):

    try:
        inner_uid = trainset.to_inner_uid(user_id)
        seen_inner = trainset.ur[inner_uid]
        seen_movie_ids = {
            int(trainset.to_raw_iid(inner_iid))
            for (inner_iid, _) in seen_inner
        }
    except ValueError:
        seen_movie_ids = set()

    candidates = [
        (
            int(mid),title, weight, poster_path) for mid, title, weight, poster_path in zip(
            item_stats["movieId"],
            item_stats["title"],
            item_stats["cult_weight"],
            item_stats.get("poster_path", [None]*len(item_stats))
        )
        if mid not in seen_movie_ids
    ]

    preds = []
    for movie_id, title, weight, poster_path in candidates:
        raw_pred = algo.predict(uid=user_id, iid=movie_id).est
        final_score = raw_pred *weight
        preds.append((title, movie_id, raw_pred, weight, final_score, poster_path))

    preds.sort(key=lambda x: x[4], reverse=True)

    return pd.DataFrame(preds[:top_n],
                        columns=[
                            "title",
                            "movieId",
                            "predicted_rating",
                            "cult_weight",
                            "final_score",
                            "posert_path"]
                        )
# 5. Demo for a random user

if __name__ == '__main__':
    algo, rmse, mae = train_SVD_and_eval(trainset,testset)
    
    sample_user = int(np.random.choice(ratings['userId'].unique()))

    print(f"\n=== Recommendations for User {sample_user} ===")

    recs = recommend_cult_classics(algo,trainset,sample_user,top_n=15)
    print(recs[['title','predicted_rating','cult_weight','final_score']]
              .to_string(index=False, float_format='%.3f'))

def build_dummy_user_model(seed_movie_ids, n_factors=N_FACTORS, n_epochs=N_EPOCHS, lr_all=LR_ALL, reg_all=REG_ALL):
    data_dict = load_movielens_data(DATA_DIR / "ml-latest-small")
    ratings = data_dict['ratings']
    dummy_uid = 999999
    
    extra = pd.DataFrame([
        [dummy_uid, mid, 5.0] for mid in seed_movie_ids
    ], columns = ["userId", "movieId", "rating"])
    
    df_with_dummy = pd.concat([ratings, extra], ignore_index=True)
    df_with_dummy = df_with_dummy[["userId", "movieId", "rating"]].astype({"userId": int, "movieId": int, "rating": float})
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(df_with_dummy, reader)
    trainset = data.build_full_trainset()
    
    algo = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all =lr_all, reg_all =reg_all,random_state=42)
    algo.fit(trainset)
    
    return algo, trainset