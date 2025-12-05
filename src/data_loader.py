import os
import pandas as pd

folder_path = "../data/ml-latest-small"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_movielens_data(folder_path):
    dataframes = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            key = filename.replace('.csv', '')
            file_path = os.path.join(folder_path, filename)
            try:
                dataframes[key] = pd.read_csv(file_path)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    return dataframes

def print_dataset_summary(dataframes):
    for key, df in dataframes.items():
        print(f"{key}: {df.shape[0]} rows, {df.shape[1]} columns")
        print(df.head())

if __name__ == "__main__":
    data = load_movielens_data(folder_path)
    print(f"Loaded {len(data)} files.")
    print_dataset_summary(data)

    # ✅ Ensure processed folder exists
    os.makedirs(os.path.join(SCRIPT_DIR, "..", "data", "processed"), exist_ok=True)

    # ✅ Save cleaned ratings file
    ratings = data["ratings"]
    ratings.to_csv(os.path.join(SCRIPT_DIR, "..", "data", "processed", "ratings_clean.csv"), index=False)
    print("Saved cleaned ratings to data/processed/ratings_clean.csv")

    # ✅ Save cleaned movies file
    movies = data["movies"]
    movies.to_csv(os.path.join(SCRIPT_DIR, "..", "data", "processed", "movies_clean.csv"), index=False)
    print("Saved cleaned movies to data/processed/movies_clean.csv")
