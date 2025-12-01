#genre_balanced_seletor.py
# provides 50 movies from the top 500 balanced based on each genre's representation in the top 200

import pandas as pd
import numpy as np
from pathlib import Path
import tmdbsimple as tmdb
import json
from dotenv import load_dotenv
import os
from data_loader import load_movielens_data
load_dotenv()

tmdb.API_KEY= os.getenv("TMDB_API_KEY")

TMDB_GENRE_MAP = {
	28: "Action", 12:"Adventure", 16: "Animation", 35: "Comedy", 80:"Crime", 99:"Documentary", 18: "Drama", 10751: "Family", 14: "Fantasy", 36: "History", 27: "Horror", 10402: "Music", 9648: "Mystery", 10749: "Romance", 878: "Science Fiction", 10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western"
}

def get_top200_genre_ratios():
	genre_count = {name:0 for name in TMDB_GENRE_MAP.values()}
	total_tags = 0

	for page in range(1,11): 
		resp = tmdb.Movies().top_rated(page=page, language="en")
		for movie in resp["results"]: 
			details = tmdb.Movies(movie["id"]).info()
			for genre in details.get("genres", []):
				name = genre["name"]
				if name in genre_count:
					genre_count[name]+=1
					total_tags+=1

	ratios = {g: c / total_tags for g, c in genre_count.items()}
	return ratios

def get_balanced50_movies():
	data = load_movielens_data("ml-latest-small")
	movies_df = data["movies"]
	links_df = data["links"]

	movies_df["genres_list"] = movies_df["genres"].str.split("|")

	ratios = get_top200_genre_ratios()

	target_n = 200
	selected_records=[]

	for genre, ratio in ratios.items():
		target_count = max(1, round(ratio*target_n))
		candidates= movies_df["genres_list"].apply(lambda x:genre in x if x else False)
		if len(candidates)==0:
			continue
		
		selected_records.extend(movies_df[movies_df["genres_list"].str.contains(genre, na=False)].to_dict(orient="records"))

	selected_df = pd.DataFrame(selected_records)
	if len(selected_df) < target_n:
		already_ids = selected_df["movieId"].tolist() if len(selected_df) > 0 else []
		remaining = movies_df[~movies_df["movieId"].isin(already_ids)]
		needed = target_n - len(selected_df)
		fillers = remaining.sample(n=needed, random_state = 49)
		selected_df = pd.concat([selected_df, fillers], ignore_index=True)

	selected_df = selected_df.sample(frac=1, random_state=49). head(target_n).copy()
	final_df = selected_df.merge(
		links_df[["movieId", "tmdbId",]],
		on="movieId",
		how="left",
	)
	final_df["tmdbId"] =final_df["tmdbId"].astype("float64")
	return final_df[["movieId", "title", "tmdbId", "genres"]]
CACHE_FILE = Path("data/balanced_movies_cache.pkl")
if CACHE_FILE.exists():
	balanced_movies_df = pd.read_pickle(CACHE_FILE)
	print("Loading balanced movies from cache")
else:
	print("Computing balanced movies...")
	balanced_movies_df = get_balanced50_movies()
	CACHE_FILE.parent.mkdir(exist_ok=True)
	balanced_movies_df.to_pickle(CACHE_FILE)
	print("Balanced movies caached for future runs")
balanced_movies_df=get_balanced50_movies()