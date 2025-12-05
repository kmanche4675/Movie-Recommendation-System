# recommender_engine.py

import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import pickle

from data_loader import load_movielens_data
from cult_classic_recommender import *
from genre_balanced_selector import balanced_movies_df
from content_based import(
    get_unique_movie_data,
    build_and_save_cbf_model,
    load_cbf_artifacts,
    get_cbf_recommendations,
)

data = load_movielens_data(Path(__file__).parent.parent / "data" / "ml-latest-small")
ratings_df = data["ratings"]
movies_df = data["movies"]
links_df = data["links"]
tags_df = data.get("tags")


#
# Classic Collaborative
#

classic_algo = None

def ensure_classic_model():
	global classic_algo
	path = Path(__file__).resolve().parent.parent / "models" / "classic_svd_best.pkl"
	if path.exists():
		classic_algo = joblib.load(path)
		print("Classic model loaded")
		return
	
	print("training classic SVD model...")
	ratings = ratings_df.copy()
	reader = Reader(rating_scale=(0.5,5.0))
	surprise_data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
	trainset = surprise_data.build_full_trainset()

	algo = SVD(n_factors=100, n_epochs=20, lr_all=0.007, reg_all=0.02, random_state=42)
	algo.fit(trainset)

	path.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(algo, path)
	classic_algo = algo
	print("Classic model trained and saved")
	
ensure_classic_model()
def filter_seeds_in_trainset(seeds, algo):
	valid_seeds = []
	for mid in seeds:
		try:
			algo.trainset.to_inner_iid(st(mid))
			valid_seeds.append(mid)
		except ValueError:
			pass
	return valid_seeds
	
def classic_recommend(seeds: list[int], top_n: int=20):
	"""Traditional Collaborative filtering (no obscurity boost)"""
	if classic_algo is None:
		raise RuntimeError("Classic model not initialized")
	seed_vectors =[]
	for mid in seeds:
		try:
			#st.write(f"Trying seed movieId: {mid}")
			iid = classic_algo.trainset.to_inner_iid(mid)
			#st.write(f" matched to inner iid: {iid}")
			seed_vectors.append(classic_algo.qi[iid])
		except ValueError:
			#st.write(f"seed {mid} not in trainset")
			continue
	if not seed_vectors:
		print("No seed movies found in training set using global average item vector")
		avg_seed_vector = np.mean(classic_algo.qi, axis=0)
	else:
		avg_seed_vector = np.mean(seed_vectors, axis=0)

	recommendations = []
	for iid in range(classic_algo.trainset.n_items):
		raw_mid = int(classic_algo.trainset.to_raw_iid(iid))
		if raw_mid in seeds:
			continue
		similarity = float(np.dot(avg_seed_vector, classic_algo.qi[iid]))
		recommendations.append((raw_mid, similarity))
		
	recommendations.sort(key=lambda x: x[1], reverse=True)

	result = []
	for mid, sim_score in recommendations[:top_n]:
		title_row = movies_df[movies_df["movieId"] == mid]["title"]
		title = title_row.iloc[0] if not title_row.empty else "Unknown Title"
		result.append({
			"movieId": mid,
			"title": title,
			"score": round(float(sim_score), 4)
			})
	return result

#
# Cult Classic predictor
#
def cult_recommend(seeds: list[int], top_n: int = 20):
	"""Cult Classic recommender: Collaborative with obscurity bias"""
	from cult_classic_recommender import build_dummy_user_model, recommend_cult_classics
	algo, trainset = build_dummy_user_model(seeds)
	
	recs = recommend_cult_classics(algo, trainset, user_id=999999, top_n=top_n +len(seeds))
	recs = recs[~recs["movieId"].isin(seeds)]
	
	recs = recs.merge(item_stats[["movieId", "tmdbId"]], on="movieId", how="left")
	result = []
	for _, row in recs.head(top_n).iterrows():
		result.append({
			"movieId": int(row["movieId"]),
			"title": row["title"],
			"score": float(row["final_score"]), 
			"tmdbId": int(row["tmdbId"]) if not pd.isna(row["tmdbId"]) else None
		})
	return result

# Content-Based (TF_IDF on genres + tags)
#

# i recommend setting up a temp_user like mine
# result =[] of top 20 recommendations from your predictor
def ensure_cbf_model():
    cosine_sim, indicies, movies_df = load_cbf_artifacts()
    if cosine_sim is not None and movies_df is not None:
        return
    movies_df_unique = get_unique_movie_data()
    if movies_df_unique is None:
        raise RuntimeError(
			"Content-based model cannot be built. "
			"MERGED_DATA_FILE is missing. Run data_loader first."
		)
    build_and_save_cbf_model(movies_df_unique)
    
def content_recommend(seeds:list[int], top_n: int = 20):
    ensure_cbf_model()
    recs = get_cbf_recommendations(seeds, top_n=top_n)
    return recs
#
# Hybrid
#
movies_hybrid = movies_df.copy()

if tags_df is not None:
	tags_df = tags_df.groupby("movieId")["tag"].apply(lambda x: " ".join(x)).reset_index()
	movies_hybrid = movies_hybrid.merge(tags_df, on="movieId", how="left")
	movies_hybrid["tag"] = movies_hybrid["tag"].fillna("")
else:
	movies_hybrid["tag"] = ""

movies_hybrid['features'] = movies_hybrid['genres'].fillna('') + ' ' + movies_hybrid['tag'].str.strip()
id_to_row = {int(row.movieId): idx for idx, row in movies_hybrid[["movieId"]].iterrows()}

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_hybrid['features'])

# Collaborative Filtering Pre-processing
user_item = ratings_df.pivot_table(index="userId", columns="movieId", values="rating")
user_item_centered = user_item.sub(user_item.mean(axis=1), axis=0).fillna(0)
item_vectors = user_item_centered.values.T
collab_sim = cosine_similarity(item_vectors, item_vectors)
movie_ids = user_item_centered.columns.to_numpy()
movieId_to_collab_idx = {int(mid): i for i, mid in enumerate(movie_ids)}

def hybrid_recommend(seeds: list[int], top_n: int =  20):
	if not seeds:
		raise ValueError("No seed movies provided.")
	
	alpha = 0.6

	# Content-based profile
	seed_idxs = []
	for mid in seeds:
		matches = movies_hybrid.index[movies_hybrid["movieId"] == mid].tolist()
		if matches:
			seed_idxs.append(matches[0])
	if not seed_idxs:
		raise ValueError("No matching seed movies found in content profile.")
	
	content_user_vector = tfidf_matrix[seed_idxs].mean(axis=0)
	content_user_vector = np.asarray(content_user_vector).reshape(1,-1)
	content_scores = cosine_similarity(content_user_vector, tfidf_matrix).ravel()

	# Collaborative profile
	collab_idxs = [movieId_to_collab_idx[mid] for mid in seeds if mid in movieId_to_collab_idx]
	if collab_idxs:
		collab_user_vector = collab_sim[collab_idxs].mean(axis=0)
		collab_scores = pd.Series(0.0, index=movies_hybrid.index)
		for i in range(len(movie_ids)):
			mid = int(movie_ids[i])
			row_idx = id_to_row.get(mid)
			if row_idx is not None:
				collab_scores.iloc[row_idx] = float(collab_user_vector[i])
		collab_scores = collab_scores.to_numpy()
	else:
		collab_scores = 0.0 * content_scores

	# Combine scores
	hybrid_scores = alpha * content_scores + (1 - alpha) * collab_scores

	exclude_ids = set(seeds)
	candidates = pd.DataFrame({
		"movieId": movies_hybrid["movieId"],
		"title": movies_hybrid["title"],
		"genre": movies_hybrid["genres"],
		"score": hybrid_scores
	})
	candidates = candidates[~candidates["movieId"].isin(exclude_ids)]
	candidates = candidates.sort_values(by="score", ascending=False).head(top_n)

	result = []
	for _, row in candidates.iterrows():
		result.append({
			"movieId": int(row["movieId"]),
			"title": row["title"],
			"score": round(float(row["score"]), 4)
		})
	return result

engine ={
	"classic": classic_recommend,
	"classic_collaboration": classic_recommend,
	"cult": cult_recommend, 
	"content": content_recommend,
	"hybrid": hybrid_recommend,
}
def predict(predictor_name: str, seed_movie_ids: list[int], top_n: int = 20):
	func = engine.get(predictor_name)
	if func is None:
		raise ValueError(f"Unknonw predictor: {predictor_name}")
	return func(seed_movie_ids, top_n=top_n)
	

if __name__ == "__main__":
	print("\n=== Testint Classic Model ===")
	test_seeds =[1,260,1196,1210,589]
	recs =classic_recommend(test_seeds, topn=10)
	print(f"Got {len(recs)} recommendsations")
	for r in recs[:5]:
		print(f"{r['title']:50} | score: {r['score']:.4f}")
	print("=== Test Done ===")
	