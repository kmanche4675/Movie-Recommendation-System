# Movie-Recommendation-System
This project builds a movie recommendation system using Machine Learning & Data Analysis principles as well as the [MovieLens dataset](https://grouplens.org/datasets/movielens). It predicts movies a user is likely to enjoy based on past ratings.

## Features

- User chooses movies that they have previously enjoyed
- Movie recommendation system uses content-based filtering, collaborative-based filtering, or a content-collaborative hybrid-based filtering to recommend movies that the user might have seen
- User rates the recommended movies on a scale of 1-5 to finetune the model
- Model generates movies that the user might want to see

# Getting Started

### Clone the Repo

```bash
git clone https://github.com/kmanche4675/Movie-Recommendation-System.git
cd Movie-Recommendation-System
```

### Downloading the Dataset

To download and extract the MovieLens dataset (ml-latest-small), run:

```bash
python src/download_movielens.py
```
### Run the SVD tuner

Optuna tuner currently set to 120 cycles. This takes a while. reduce in code for faster run. This should create a models folder to save model and best parameters. To run the tuner run:

```bash
python src/cult_recommender_tuner.py
```

### Produce genre balanced list of movies

genre_balanced_selctor.py currently produces 75 movies balanced by genre as genres are represented in the top 200 movies. If you want more or less to choose from, this is the file to edit. To produce initial selection list, run:
```bash
python src/genre_balanced_selector.py
```

### Get movie posters

To get the movie posters for the 75 selected from genre balaced selection and the rest of the MovieLens database, run:
```bash
python src/get_the_proper_posters.py
```

### Launch the streamlit application

Launch the streamlit application from the root directory. To launch run:
```bash
streamlit run src/streamlit_cult_classic_recommender.py
```

Select some movies you like. The recommender will treat your movies as 5 star ratings. Choose from pure classic collaborative, content, hybrid, and cult classic filtering. Enjoy!
