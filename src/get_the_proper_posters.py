import pandas as pd
import tmdbsimple as tmdb
import os
from pathlib import Path
from dotenv import load_dotenv
import requests
from PIL import Image
from io import BytesIO
import time
import requests
from data_loader import load_movielens_data

load_dotenv()
tmdb.API_KEY = os.getenv("TMDB_API_KEY")
if not tmdb.API_KEY:
	raise RuntimeError("TMDB_API_KEY not found in .env")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = PROJECT_ROOT / "balanced_movies.csv"
POSTER_FOLDER = PROJECT_ROOT / "posters"
OUTPUT_CSV = PROJECT_ROOT / "balanced_movies_with_posters.csv"

POSTER_FOLDER.mkdir(exist_ok=True, parents=True)

df =pd.read_csv(DATA_CSV)
if "poster_path_local" in df.columns:
    poster_paths = df["poster_path_local"].tolist()
else:
    poster_paths = [None]* len(df)
    
def download_image(url, save_path: Path, retries: int=10, delay: float = 1.0) -> bool:
    for attempt in range(1, retries +1):
        try:
            #print(f"Trying {url}")
            resp = requests.get(url, timeout=10)
			#print(f"Status {resp.status_code}")
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            img.save(save_path, format = "JPEG")
			#print(f"Saved {save_path}")
            return True
        except Exception as e:
            print(f" [download failed attempt {attempt}/{retries}] {save_path.name}: {e}")
            if attempt < retries:
                time.sleep(delay)
    return False

def get_tmdb_details(tmdb_id=None, title=None, year=None):
	if tmdb_id is not None:
		try:
			details =  tmdb.Movies(int(tmdb_id)).info()
			if details.get("poster_path"):
				return details
		except Exception:
			pass
	if title:
		try:
			search = tmdb.Search()
			results = search.movie(query=title, year=year)
			if results["results"]:
				return results["results"][0]
		except Exception:
			pass
	return None
poster_paths=[]*len(df)

for idx, row in df.iterrows():
	title = row["title"] 
	year = None

	if "(" in title and ")" in title:
		try: 
			year = int(title.split("(")[-1].replace(")",""))
		except: 
			pass
		tmdb_id = row.get("tmdbId")
		details = get_tmdb_details(tmdb_id=tmdb_id, title=title, year=year)

		if details and details.get("poster_path"):
			poster_url = f"https://image.tmdb.org/t/p/w500{details['poster_path']}"
			filename = f"{details['id']}.jpg"
			local_path = POSTER_FOLDER / filename
			success = download_image(poster_url, local_path)
			if success:
				poster_paths.append(str(local_path))
				print(f"[{idx+1}/{len(df)}] Downloaded: {title}")
				time.sleep(0.25)
			else:
				poster_paths[idx] = ModuleNotFoundError
df["poster_path_local"] = poster_paths
df.to_csv(OUTPUT_CSV, index= False)
print("Saved enriched CSV with local poster paths.")

print("\nBuilding posters for ALL MovieLens movies (needed for recommendations)...")

# Load ALL MovieLens movies
data_full = load_movielens_data(Path(__file__).parent.parent / "data" / "ml-latest-small")
movies_full = data_full["movies"]
links_full = data_full["links"].dropna(subset=["tmdbId"]).copy()
links_full["tmdbId"] = links_full["tmdbId"].astype(int)

full_df = movies_full.merge(links_full[["movieId", "tmdbId"]], on="movieId", how="inner")
full_df["poster_path_local"] = None

total_full = len(full_df)
success_full = 0
print(f"Found {total_full} movies with tmdbId. Downloading posters...")

for idx, row in full_df.iterrows():
    tmdb_id = row["tmdbId"]
    title = row["title"]

    print(f"[{idx+1}/{total_full}] {title}")

    try:
        details = tmdb.Movies(int(tmdb_id)).info()
    except Exception as e:
        print(f"  TMDB error: {e}")
        continue

    poster_path = details.get("poster_path")
    if not poster_path:
        print("  No TMDB poster")
        continue

    filename = f"{details['id']}.jpg"
    local_path = POSTER_FOLDER / filename
    url = f"https://image.tmdb.org/t/p/w500{poster_path}"

    if not local_path.exists():
        ok = download_image(url, local_path)
        if ok:
            print("  Downloaded poster")
            success_full += 1
            time.sleep(0.2)
        else:
            print("  Failed download")
            continue
    else:
        print("  Already exists")

    full_df.at[idx, "poster_path_local"] = str(local_path)

full_df.to_csv(OUTPUT_CSV_ALL, index=False)
print(f"\nSaved full poster map to: {OUTPUT_CSV_ALL}")
print(f"Posters downloaded (new): {success_full}/{total_full}")