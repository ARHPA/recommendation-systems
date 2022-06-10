import pandas as pd
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

# Storing the movie information
movies_df = pd.read_csv('movies.csv')
# Storing the user information
ratings_df = pd.read_csv('ratings.csv')

# Using regular expressions to find a year stored between parentheses
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))', expand=False)
# Removing the parentheses
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)', expand=False)
# Removing the years from the 'title' column
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
# Applying the strip function to get rid of any ending whitespace characters
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())

movies_df['genres'] = movies_df.genres.str.split('|')

# Copying the movie dataframe into a new one since
moviesWithGenres_df = movies_df.copy()

for index, row in movies_df.iterrows():
    for genre in row['genres']:
        moviesWithGenres_df.at[index, genre] = 1
# Filling in the NaN values with 0
moviesWithGenres_df = moviesWithGenres_df.fillna(0)

ratings_df = ratings_df.drop('timestamp', 1)

userInput = [
    {'title': 'Breakfast Club, The', 'rating': 5},
    {'title': 'Toy Story', 'rating': 3.5},
    {'title': 'Jumanji', 'rating': 2},
    {'title': "Pulp Fiction", 'rating': 5},
    {'title': 'Akira', 'rating': 4.5}
]
inputMovies = pd.DataFrame(userInput)

# Filtering out the movies by title
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
# Then merging it so we can get the movieId
inputMovies = pd.merge(inputId, inputMovies)

inputMovies = inputMovies.drop('genres', 1).drop('year', 1)

# Filtering out the movies from the input
userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]

userMovies = userMovies.reset_index(drop=True)

userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)

userProfile = userGenreTable.transpose().dot(inputMovies['rating'])
# The user profile

genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])
# And drop the unnecessary information
genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)

recommendationTable_df = ((genreTable * userProfile).sum(axis=1)) / (userProfile.sum())
recommendationTable_df.head()

# Sort our recommendations in descending order
recommendationTable_df = recommendationTable_df.sort_values(ascending=False)

# The final recommendation table
print(movies_df.loc[movies_df['movieId'].isin(recommendationTable_df.head(20).keys())])
