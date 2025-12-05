# cult_recommender_tuner.py
import optuna
from optuna import visualization as vis
from optuna.trial import TrialState
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
import numpy as np
import pandas as pd
import json
import math
from pathlib import Path
from data_loader import load_movielens_data
import requests
import os
import time
from dotenv import load_dotenv
import joblib

load_dotenv()
TMDB_API_KEY=os.getenv("TMDB_API_KEY")
if not TMDB_API_KEY:
	raise ValueError("TMDB_API_KEY not found in .env file")
	
# Load MovieLens data
print("loading MovieLens data...")
data_dict = load_movielens_data("./data/ml-latest-small")
ratings = data_dict['ratings']
movies = data_dict['movies']
links = data_dict['links'].dropna(subset = ['tmdbId'])
links['tmdbId'] = links['tmdbId'].astype(int)
# load TMDB data
print("loading TMDB metadata...")
tmdb_records = []

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_PATH = PROJECT_ROOT / "models"
TMDB_CACHE_FILE= DATA_DIR / "tmdb_metadata.csv"

MODELS_PATH.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

def get_tmdb_data(tmdb_id):
	url=f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}"
	for attempt in range(3):
		try:
			r = requests.get(url, timeout=10)
			if r.status_code ==200:
				data = r.json()
				return{
					'tmdbId': data['id'],
					'vote_average': data.get('vote_average', 0.0),
					'vote_count': data.get('vote_count',0),
					'popularity': data.get('popularity',0.0)
				}
			elif r.status_code ==429:
				time.sleep(2)
		except requests.exceptions.RequestException:
			time.sleep(1)
	return {'tmdbId': tmdb_id, 'vote_average': 0.0, 'vote_count': 0, 'popularity': 0.0}
	 
tmdb_records =[]
unique_tmdb_ids = links['tmdbId'].astype(int).unique()


if os.path.exists(TMDB_CACHE_FILE):
	print("Loading cached TMDB metadata...")
	tmdb_df = pd.read_csv(TMDB_CACHE_FILE)
else:
	print("Fetching TMDB metadata via API...")
	
	for i, tmdb_id in enumerate(unique_tmdb_ids):
		if i% 500 ==0:
			print(f"fetched: {i}/{len(unique_tmdb_ids)} movies...")
		tmdb_records.append(get_tmdb_data(tmdb_id))
		time.sleep(0.11)
		
	tmdb_df  = pd.DataFrame(tmdb_records)
	tmdb_df.to_csv(TMDB_CACHE_FILE, index =False)
	print(f"Saved TMDB metadata cache: {TMDB_CACHE_FILE}")
	
print(f"TMDB dataframe ready: {len(tmdb_df):,} movies")

# merge datasets
item_stats = links.merge(tmdb_df, on='tmdbId', how='left').merge(movies[['movieId','title']], on='movieId')

#Surprise dataset
reader = Reader(rating_scale=(0.5,5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

#
#Classic tuner
#

def tune_classic():
	print("\nTuning Classic SVD multi-objective RMSE and MAE...")
	def objective(trial):
		params = {
			"n_factors": trial.suggest_int("n_factors", 80, 200),
			"n_epochs": trial.suggest_int("n_epochs", 20, 50),
			"lr_all": trial.suggest_float("lr_all", 0.002, 0.015, log = True),
			"reg_all": trial.suggest_float("reg_all", 0.02, 0.2, log = True),
										   }
		algo = SVD(**params, random_state=42)
		trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
		algo.fit(trainset)

		predictions = algo.test(testset)
		rmse = accuracy.rmse(predictions, verbose=False)
		mae = accuracy.mae(predictions, verbose=False)

		return rmse, mae

	study = optuna.create_study(directions=["minimize", "minimize"])
	study.optimize(objective, n_trials = 120, timeout = 1800)
	classic_history_df = study.trials_dataframe()
	classic_history_df.to_csv(PROJECT_ROOT / "classic_svd_optuna_history.csv", index=False)
 

	print("\nClassic SVD - Pareto Front (top5):")
	for i, trial in enumerate(study.best_trials[:5]):
		print(f"#{i+1} RMSE:{trial.values[0]:.4f} | MAE: {trial.values[1]:.4f} | {trial.params}")
	best = min(study.best_trials, key=lambda t: t.values[0])
	print(f"\nSELECTED BEST CLASSIC MODEL RMSE: {best.values[0]:.4f}, MAE: {best.values[1]:.4f}")

	final_algo = SVD(**best.params, random_state=42)
	final_algo.fit(data.build_full_trainset())

	
	MODELS_PATH.mkdir(exist_ok=True, parents=True)

	
	joblib.dump(final_algo, MODELS_PATH / "classic_svd_best.pkl")
	print(f"Saved {MODELS_PATH / 'classic_svd_best.pkl'}")

	classic_config = {
		"model_type": "Classic SVD",
		"objective": "Minimize RMSE and MAE",
		"best_rmse": round(best.values[0],6),
		"best_mae": round(best.values[1],6),
		"hyperparameters":{
			"n_factors": best.params["n_factors"],
			"n_epochs": best.params["n_epochs"],
			"lr_all": best.params["lr_all"],
			"reg_all": best.params["reg_all"]
		},
		"Note": "Trained on MovieLens Dataset with multi_objective optimization"
	}
	
	with open(MODELS_PATH / "classic_svd_best.json", "w") as f:
		json.dump(classic_config, f, indent=2)
	print(f"Saved {MODELS_PATH}/classic_svd_best.json")
	

	try:
		fig = vis.plot_pareto_front(study, target_names= ["RMSE", "MAE"])
		fig.write_image(PROJECT_ROOT / "data" / "classic_pareto_rmse_mae.png")
		print("Saved classic_pareto_rmse_mae.png")
	except:
		pass

#
#Cult Tuner
#

#optimize cult score weights
def compute_cult_weights(df, obscurity_exp, min_votes):
	def cult_score(row):
		if row['vote_count']<min_votes:
			return 0.0
		quality = row['vote_average']/10.0
		pop = max(row['popularity'],1.0)
		obscurity = 1.0/(math.log1p(pop) ** obscurity_exp)
		return quality * obscurity
	
	df['cult_weight'] = df.apply(cult_score, axis =1)
	return df[df['cult_weight']>0].copy()

def evaluate_ranking_metrics(testset, algo, item_stats_with_weights, top_n=20):
	precisions, recalls = [],[]

	user_liked={}
	for uid, mid, r in testset:
		if r>= 4.0:
			user_liked.setdefault(uid, set()).add(mid)
	for uid, liked_movies in user_liked.items():
		if not liked_movies:
			continue
		try:
			seen = {algo.trainset.to_raw_iid(iid)
					for iid, _ in algo.trainset.ur[algo.trainset.to_inner_uid(uid)]}
		except ValueError:
			continue
			
		candidates = item_stats_with_weights[~item_stats_with_weights['movieId'].isin(seen)]
		preds = [(row['movieId'], algo.predict(uid, row['movieId']).est * row['cult_weight']) for _, row in candidates.iterrows()]
		preds.sort(key=lambda x: x[1], reverse=True)
		recommended = [mid for mid, _ in preds[:top_n]]

		hits = len(set(recommended) & liked_movies)

		precision = hits/top_n
		recall = hits / len(liked_movies)

		precisions.append(precision)
		recalls.append(recall)
	if not precisions:
		return 0.0, 0.0
	return np.mean(precisions), np.mean(recalls)

	
def tune_cult():	
	print("\nTUNING CULT CLASSIC (RMSE and Precision@20)...")
	
	def objective(trial):
		n_factors = trial.suggest_int('n_factors', 50,300)
		n_epochs = trial.suggest_int('n_epochs', 20,60)
		lr = trial.suggest_float('lr_all', 5e-4,2e-2, log=True)
		reg = trial.suggest_float('reg_all', 0.005,0.25, log= True)
	
		obscurity_exp = trial.suggest_float('obscurity_exp', .3,1.7)
		min_votes = trial.suggest_int('min_votes', 80, 300)
	
		trainset, testset = train_test_split(data, test_size=.2, random_state=42)
		algo = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr, reg_all=reg, random_state=42)
		algo.fit(trainset)
	
		rmse=accuracy.rmse(algo.test(testset), verbose=False)
		weighted = compute_cult_weights(item_stats.copy(), obscurity_exp, min_votes)
		precision, _ = evaluate_ranking_metrics(testset, algo, weighted)
		return rmse, -precision
	
	study= optuna.create_study(directions=['minimize','maximize'])
	print("starting optuna tuning(120 trials)...")
	study.optimize(objective, n_trials=120)
	cult_history_df = study.trials_dataframe()
	cult_history_df.to_csv(PROJECT_ROOT / "cult_svd_optuna_history.csv", index=False)
	
	
	print("\nBest Trials (pareto front): ")
	for i, t in enumerate(study.best_trials): 
		rmse_val= t.values[0]
		precision_val = -t.values[1]
		print(f" Pareto[{i}] RMSE = {rmse_val:.6f}, Precision@20 = {precision_val:.6f}, params={t.params}")
	
	best = sorted(study.best_trials, key=lambda tr:(tr.values[0], tr.values[1]))[0]
	
	print(f"\nSelected best trial (for deployment): ")
	print(f"\nRMSE: {best.values[0]:.6f}, Precision@20: {-best.values[1]:.6f}")
	print(f"Params: {best.params}")

	final_algo = SVD(
		n_factors=best.params["n_factors"],
		n_epochs=best.params["n_epochs"],
		lr_all=best.params["lr_all"],
		reg_all=best.params["reg_all"], 
		random_state = 42
	)
	final_algo.fit(data.build_full_trainset())
	
	MODELS_PATH.mkdir(parents=True, exist_ok=True)
	
	joblib.dump(final_algo, MODELS_PATH / "cult_svd_best.pkl")
	print(f"Saved {MODELS_PATH / 'cult_svd_best.pkl'}")
	
	config= {
		"svd": {
			"n_factors": best.params["n_factors"],
			"n_epochs": best.params["n_epochs"],
			"lr_all": best.params["lr_all"],
			"reg_all": best.params["reg_all"],
		},
		"cult":{
			"obscurity_exp": best.params["obscurity_exp"],
			"min_votes": best.params["min_votes"],
		}
	}
	with open(MODELS_PATH /"cult_classic_best.json", "w") as f:
		json.dump(config, f, indent =2)
	print(f"Saved {MODELS_PATH}/cult_classic_best.json")

	try:
		fig = vis.plot_pareto_front(study, target_names=["RMSE", "Precision@20"])
		fig.write_image(PROJECT_ROOT /"data" / "cult_pareto_rmse_precision.png")
		print("Saved cult_pareto_rmse_precision.png")
	except Exception as e:
		print("Could not save plot:", e)
if __name__ == "__main__":
	tune_classic()
	tune_cult()
	print(f"\nBOTH RECOMMENDERS FULLY TUNED and SAVED in {MODELS_PATH}")
			