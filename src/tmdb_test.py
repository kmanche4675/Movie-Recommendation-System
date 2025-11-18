###test script for TMDB to test access with your api key Code is ChatGPT written
###Script shows structure for requests using the API
import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

if not TMDB_API_KEY:
    raise ValueError("TMDB_API_KEY not found in .env")

# Example endpoint: List movie genres
url = "https://api.themoviedb.org/3/genre/movie/list"

params = {
    "api_key": TMDB_API_KEY,
    "language": "en-US"
}

response = requests.get(url, params=params)

if response.status_code == 200:
    data = response.json()
    print("\n✅ API connected successfully!")
    print("Returned keys:", list(data.keys()))
    print("\nSample data:\n", data)
else:
    print(f"\n❌ Request failed with status {response.status_code}")
    print(response.text)
