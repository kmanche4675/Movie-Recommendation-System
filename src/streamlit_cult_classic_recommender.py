# streamlit_cult_classic.py
import sys
import os

conda_env_path = os.environ.get("CONDA_PREFIX")
if conda_env_path:
	site_packages = f"{conda_env_path}/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
	if site_packages not in sys.path:
		sys.path.insert(0,site_packages)
		
import streamlit as st
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv
import tmdbsimple as tmdb
from genre_balanced_selector import balanced_movies_df
import json
from recommender_engine import *
import traceback


load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
if not TMDB_API_KEY:
	st.error("TMDB_API_KEY not found. Create .env file with your key.")
	st.stop()
	
tmdb.API_KEY = TMDB_API_KEY

POSTER_FOLDER = Path("posters")
POSTER_FOLDER.mkdir(exist_ok=True)

PLACEHOLDER_NO_POSTER = "https://placehold.co/500x750/cccccc/666666/png?text=No+Poster"
PLACEHOLDER_404 = "https://placehold.co/500x750/E74C3C/FFFFFF/png?text=404"

def _download_image_to_pil(url, save_to:Path |None = None, timeout=6):
	
	try:
		resp = requests.get(url, timeout=timeout)
		resp.raise_for_status()
		img = Image.open(BytesIO(resp.content)).convert("RGB")
		if save_to:
			try:
				img.save(save_to, format="JPEG")
			except Exception:
				pass
		return img

	except Exception:
		return None
		
@st.cache_data(ttl=3600)
def get_poster(tmdb_id=None, poster_url=None):

	if poster_url:
		img = _download_image_to_pil(poster_url, save_to=(POSTER_FOLDER / f"{int(tmdb_id)}.jpg") if tmdb_id else None)
		if img:
			return img
	if not tmdb_id or pd.isna(tmdb_id):
		img = _download_image_to_pil(PLACEHOLDER_NO_POSTER)
		return img if img else Image.new("RGB", (500,750), color=(204,204,204))
	try:
		tmdb_int=int(tmdb_id)
	except Exception:
		tmdb_int = None
	if tmdb_int:
		cached= POSTER_FOLDER / f"{tmdb_int}.jpg"
		if cached.exists():
			try:
				return Image.open(cached).convert("RGB")
			except Exception:
				pass
		try:
			details = tmdb.Movies(tmdb_int).info()
			poster_path = details.get("poster_path")
		except Exception:
			poster_path = None

		if poster_path:
			url = f"https://image.tmdb.org/t/p/w500{poster_path}"
			img = _download_image_to_pil(url, save_to=cached)
			if img:
				return img

	img = _download_image_to_pil(PLACEHOLDER_404)
	return img if img else Image.new("RGB", (500,750), color=(231, 76, 60))

if "selected" not in st.session_state:	
    st.session_state.selected = []
if "chosen_predictor" not in st.session_state:
    st.session_state.chosen_predictor = None
    st.session_state.predictor_label = None
if "current_recs" not in st.session_state:
    st.session_state.current_recs = []
if "ratings_buffer" not in st.session_state:
    st.session_state.ratings_buffer = {}
st.set_page_config(page_title="Cult vs Classic Recommender", layout = "wide")
st.title("Cult Classic Vs Mainstream Reommender")
st.markdown(" *** Select 10-15 movies you love to get personalized recommendations ***")
st.info("""
**Selction Grid**: 50 Movies with genres weighted to match distribution of top 200
""")
				 	 
popular = balanced_movies_df.copy()
try:
	if "tmdbId" in popular.columns:
		tmdb_map = dict(zip(popular["movieId"].astype(int), popular["tmdbId"]))
	else:
		tmdb_map = dict()
except Exception:
	tmdb_map = dict()

cols = st.columns(6)
selected_movie_ids = st.session_state.selected

LINE_HEIGHT_PX = 20
MAX_TITLE_LINES = 5
TITLE_HEIGHT = MAX_TITLE_LINES * LINE_HEIGHT_PX

for idx, row in popular.iterrows():
	with cols[idx %6]:
		movie_id = int(row["movieId"])
		title = row["title"].split("(")[0].strip()
		poster = get_poster(row.get("tmdbId"))
		clicked = st.button(
			"",
			key=f"btn_{movie_id}",
			help=title,
			on_click=None
		)
		st.image(poster, width="stretch")
		if clicked:
			if movie_id in selected_movie_ids:
				selected_movie_ids.remove(movie_id)
			else:
				selected_movie_ids.append(movie_id)
			st.session_state.selected = selected_movie_ids
		if movie_id in selected_movie_ids:
			st.success("Selected")
		st.markdown(
			f'<div style="height:{TITLE_HEIGHT}px; overflow:hidden;">{title}</div>',unsafe_allow_html=True
		)
def call_predictor(name, seeds, top_n=20):
	try:
		from recommender_engine import engine
		recs = predict(name, seeds, top_n=top_n)
		return recs
	except Exception as e:
		tb = traceback.format_exc()
		st.error(f"Failed to call predict() -falling back")
		st.code(tb)
		if name == "classic":
			try:
				from recommender_engine import classic_model
				df = classic_model.recommend(seeds, top_n=top_n)
				out = []
				for _, r in df.iterrows():
					out.append({"movieId": int(r["movieId"]), "title": r["title"],"score": float(r.get("final_score", r.get("predicted", 0.0)))})
				return out
			except Exception:
				pass
		try:
			stub = popular.head(top_n)
			out = []
			for _, r in stub.iterrows():
				out.append({"movieId": int(r["movieId"]), "title": r["title"], "score": 0.0})
			return out
		except Exception as e2:
			st.error(f"Fallback classic_model.recommend also failed: {e2}")
			st.code(traceback.format_exc())
		stub = popular.head(top_n)
		out = []
		for _, r in stub.iterrows():
			out.append({
       			"movieID": int(r["movieId"]),
          		"title": r.get("title", "<no title>"),
            	"score": 0.0
             })
		#st.write("DEBUG out sample:", out[:3])
		return out
st.markdown("---")
if st.button("Submit Ratings and Log Run", width="stretch"):
	ratings = {}
	hits = 0
	k = len(st.session_state.current_recs)

	for rec in st.session_state.current_recs:
		mid = int(rec["movieId"])
		user_rating = st.session_state.ratings_buffer.get(str(mid),3)
		ratings[mid] = user_rating
		if user_rating >= 4:
			hits += 1
			
	precision_at_20 = hits / k if k >0 else 0.0

	run = {
		"timestamp": pd.Timestamp.utcnow().isoformat(),
		"predictor": st.session_state.chosen_predictor,
		"seed_movie_ids": selected_movie_ids,
		"recommendations": [
			{
				"movieId": int(r["movieId"]),
				"title": r["title"],
				"score": float(r["score"])
			}
			for r in st.session_state.current_recs
		],
		"user_ratings": ratings,
		"precision_at_20": precision_at_20,
	}

	logs_path = Path("logs")
	logs_path.mkdir(exist_ok=True)
	out_path = logs_path / "prediction_runs.ndjson"
	try:
		with open(st.session_state.log_file,"a", encoding="utf-8") as f:
			f.write(json.dumps(run, ensure_ascii=False) +"\n")
		st.success(f"Run Logged! presision@20 = {precision_at_20: .3f}")
	except Exception as e:
		st.error(f"Failed to write log: {e}")
	try:
		df_log = pd.DataFrame(run["recommendations"])
		df_log["user_rating"] = df_log["movieId"].astype(str).map(ratings).fillna(0).astype(int)
		st.dataframe(df_log[["movieId", "title", "score", "user_rating"]].head(20), width="stretch",hide_index=True)
	except Exception:
		pass
				
st.markdown("---")
st.markdown("### Chose a Recommender###")

c1, c2, c3, c4 =st.columns(4)

if "current_recs" not in st.session_state:
	st.session_state.current_recs=[]
if "chosen_predictor" not in st.session_state:
	st.session_state.chosen_predictor = None
	st.session_state.predictor_label = None

with c1:
	if c1.button("Classic Collaborative", type="primary", width="stretch", key="btn_classic"):
		st.session_state.chosen_predictor = "classic"
		st.session_state.predictor_label = "Classic Collaborative"
with c2:
	if c2.button("Content-Based", type="primary", width="stretch",key="btn_content"):
		st.session_state.chosen_predictor = "content"
		st.session_state.predictor_label = "Content-Based"
with c3:
	if c3.button("Hybrid", type="primary", width="stretch",key="btn_hybrid"):
		st.session_state.chosen_predictor = "hybrid"
		st.session_state.predictor_label = "Hybrid"
with c4:
	if c4.button("Cult Classic", type="primary", width="stretch",key="btn_cult"):
		st.session_state.chosen_predictor = "cult"
		st.session_state.predictor_label = "Cult Classic"

#trigger reommendations when a predictor is chosen
if st.session_state.chosen_predictor and selected_movie_ids:
    predictor = st.session_state.chosen_predictor
    label = st.session_state.predictor_label
    
    #st.write(f"DEBUG: Running predictor **{predictor}**")
    #st.write(f"DEBUG: Seeds {selected_movie_ids}")
    with st.spinner(f"Generating {st.session_state.predictor_label} recommendations..."):
        recs = call_predictor(st.session_state.chosen_predictor, selected_movie_ids, top_n=20)
        
    #st.write(f"DEBUG: got {len(recs)} recommendations")
    #st.write(f"DEBUG: First rec {recs[0] if recs else 'NONE'}")
    st.subheader(f"Recommendations - {st.session_state.predictor_label}")
    st.session_state.current_recs = []
    st.session_state.ratings_buffer = {}

    normalized_recs = []
    for item in recs:
        mid = int(item["movieId"])
        title = item["title"]
        score = float(item["score"])
        normalized_recs.append({"movieId": mid, "title": title, "score": score})

    st.session_state.current_recs = normalized_recs
	
    cols_per_row = 5
    for i in range(0, len(normalized_recs), cols_per_row):
        cols = st.columns(cols_per_row)
        for idx, rec in enumerate(normalized_recs[i:i+cols_per_row]):
            with cols[idx]:
                mid = rec["movieId"]
                tmdb_id = tmdb_map.get(mid)
                poster = get_poster(tmdb_id=tmdb_id) if tmdb_id else get_poster(poster_url=PLACEHOLDER_NO_POSTER)
                st.image(poster, width="stretch")
                st.markdown(f"**{rec['title']}**")
                st.markdown(f"Score: {rec['score']:.4f}")

                rating_key = f"rating_{st.session_state.chosen_predictor}_{mid}"
                initial = st.session_state.ratings_buffer.get(str(mid), 3)
                val = st.slider("Rate",1,5, initial, key=rating_key)
                st.session_state.ratings_buffer[str(mid)] = val


	
