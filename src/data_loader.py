import os
import pandas as pd

folder_path = "path/to/ml-latest-small"

def load_movielens_data(folder_path):
    """
    Reads all CSV files in the MovieLens folder and returns a dictionary of DataFrames.
    Keys are the filenames without '.csv'.
    """

    dataframes ={}
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
    """
    Prints the number of rows and columns for each DataFrame in the dictionary.
    """
    for key, df in dataframes.items():
        print(f"{key}: {df.shape[0]} rows, {df.shape[1]} columns")
        print(df.head())

if __name__ == "__main__":
    data = load_movielens_data(folder_path)
    print(f"Loaded {len(data)} files.")
    print_dataset_summary(data)
