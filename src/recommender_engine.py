# recommender_engine.py

import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from pathlib import Path
import joblib
import pickle

from data_loader import load_movielens_data
from cult_classic_recommender import *
from genre_balanced_selector import balanced_movies_df

data = load_movielens_data("ml-latest-small")
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
	path = Path("models/classic_svd.pkl")
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
			st.write(f"Trying seed movieId: {mid}")
			iid = classic_algo.trainset.to_inner_iid(mid)
			st.write(f" matched to inner iid: {iid}")
			seed_vectors.append(classic_algo.qi[iid])
		except ValueError:
			st.write(f"seed {mid} not in trainset")
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
	from cult_classic_recommender import algo, trainset
	temp_user = 999999
	df = pd.DataFrame([(temp_user, mid, 5.0) for mid in seeds],
					  columns=["userId", "movieId", "rating"])
	recs = recommend_cult_classics(algo, trainset, temp_user, top_n=top_n + len(seeds))
	recs = recs[~recs["movieId"].isin(seeds)]
	result = []
	for _, row in recs.head(top_n).iterrows():
		result.append({
			"movieId": int(row["movieId"]),
			"title": row["title"],
			"score": float(row["final_score"])
		})
	return result


#
# Content-Based (TF_IDF on genres + tags)
#

# i recommend setting up a temp_user like mine
# result =[] of top 20 recommendations from your predictor

def content_recommend(seeds:list[int], top_n: int = 20):
	return result
#
# Hybrid
#

# i recommend setting up a temp_user like mine
# result =[] of top 20 recommendations from your predictor

def hybrid_recommend(seeds: list[int], top_n: int =  20):
	return result

engine ={
	"classic": classic_recommend,
	"cult": cult_recommend, 
	"content": content_recommend,
	"hybrid": hybrid_recommend,
}
def predict(predictor_name: str, seed_movie_ids: list[int], top_n: int = 20):
	func = engine.get(predictor_name)
	if func is None:
		raise ValueError(f"Unkonw predictor: {predictor_name}")
	return func(seed_movie_ids, top_n=top_n)
	

if __name__ == "__main__":
	print("\n=== Testint Classic Model ===")
	test_seeds =[1,260,1196,1210,589]
	recs =classic_recommend(test_seeds, topn=10)
	print(f"Got {len(recs)} recommendsations")
	for r in recs[:5]:
		print(f"{r['title']:50} | score: {r['score']:.4f}")
	print("=== Test Done ===")
	
		