import os

import pandas as pd



folder_path = "./ml-latest-small"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))



def load_movielens_data(folder_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data", folder_path)
    data_dir = os.path.abspath(data_dir)

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"MovieLens dataset folder not found: {data_dir}")
    
    dataframes = {}
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            key = filename.replace('.csv', '')
            file_path = os.path.join(data_dir, filename)
            dataframes[key] = pd.read_csv(file_path)
    return dataframes

def merge_data(dataframes):
    """
    Performs critical merging and feature aggregation for ML consumption.
    This creates the comprehensive merged_data_clean.csv file.
    """
    if 'ratings' not in dataframes or 'movies' not in dataframes:
        print("ERROR: Missing core ratings or movies file for merging.")
        return None, None, None, None, None
    
    ratings_df = dataframes['ratings']
    movies_df = dataframes['movies']
    tags_df = dataframes.get('tags')
    links_df = dataframes.get('links')

    df_merged = pd.merge(ratings_df, movies_df, on='movieId', how='left')
    
    if tags_df is not None:
        tag_agg = tags_df.groupby('movieId')['tag'].apply(lambda x: ' '.join(x.astype(str))).reset_index()
        df_merged = pd.merge(df_merged, tag_agg, on='movieId', how='left')
        df_merged['tag'] = df_merged['tag'].fillna('')
    else:
        df_merged['tag'] = ''

    if links_df is not None:
         df_merged = pd.merge(df_merged, links_df, on='movieId', how='left')

    return df_merged, ratings_df, movies_df, links_df, links_df

def print_dataset_summary(dataframes):
    for key, df in dataframes.items():
        print(f"{key}: {df.shape[0]} rows, {df.shape[1]} columns")
        print(df.head())



if __name__ == "__main__":
    # Define save path (needs to be calculated first)
    PROCESSED_DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "processed")
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True) # Ensure the folder exists

    # --- STEP 1: Load Raw Data ---
    data = load_movielens_data(folder_path)
    print(f"Loaded {len(data)} raw files.")
    print_dataset_summary(data) # Keep the summary printout

    # --- STEP 2: Perform Merging (NEW CRITICAL STEP) ---
    merged_df, ratings_df, movies_df, links_df, _ = merge_data(data)
    
    if merged_df is None:
        print("Integration failed: Could not merge core data.")
        exit()
    
    # --- STEP 3: Save ALL Cleaned Files (INCLUDING THE NEW MERGED FILE) ---
    ratings = data["ratings"]
    ratings.to_csv(os.path.join(PROCESSED_DATA_PATH, "ratings_clean.csv"), index=False)
    print("Saved ratings_clean.csv (For CF/SVD Model)")

    merged_df.to_csv(os.path.join(PROCESSED_DATA_PATH, "merged_data_clean.csv"), index=False)
    print("Saved merged_data_clean.csv (For CBF/Hybrid Features)")

    movies = data["movies"]
    movies.to_csv(os.path.join(PROCESSED_DATA_PATH, "movies_clean.csv"), index=False)
    
    if links_df is not None:
        links_df.to_csv(os.path.join(PROCESSED_DATA_PATH, "links_clean.csv"), index=False)
        
    print("\n--- Data Loader Integration Complete ---")
