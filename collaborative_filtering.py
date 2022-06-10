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
# Applying the strip function
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())

# Dropping the genres column
movies_df = movies_df.drop('genres', 1)
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
# Then merging it so we can get the movieId. It's implicitly merging it by title.
inputMovies = pd.merge(inputId, inputMovies)
# Dropping information we won't use from the input dataframe
inputMovies = inputMovies.drop('year', 1)

# Filtering out users that have watched movies that the input has watched and storing it
userSubset = ratings_df[ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]

userSubsetGroup = userSubset.groupby(['userId'])

userSubsetGroup = sorted(userSubsetGroup, key=lambda x: len(x[1]), reverse=True)

pearsonCorrelationDict = {}

for name, group in userSubsetGroup:
    group = group.sort_values(by='movieId')
    inputMovies = inputMovies.sort_values(by='movieId')

    nRatings = len(group)

    temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
    tempRatingList = temp_df['rating'].tolist()
    tempGroupList = group['rating'].tolist()

    Sxx = sum([i ** 2 for i in tempRatingList]) - pow(sum(tempRatingList), 2) / float(nRatings)
    Syy = sum([i ** 2 for i in tempGroupList]) - pow(sum(tempGroupList), 2) / float(nRatings)
    Sxy = sum(i * j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList) * sum(tempGroupList) / float(nRatings)

    if Sxx != 0 and Syy != 0:
        pearsonCorrelationDict[name] = Sxy / sqrt(Sxx * Syy)
    else:
        pearsonCorrelationDict[name] = 0


pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
pearsonDF.columns = ['similarityIndex']
pearsonDF['userId'] = pearsonDF.index
pearsonDF.index = range(len(pearsonDF))


topUsers = pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]

topUsersRating = topUsers.merge(ratings_df, left_on='userId', right_on='userId', how='inner')

# Multiplies the similarity by the user's ratings
topUsersRating['weightedRating'] = topUsersRating['similarityIndex'] * topUsersRating['rating']

# Applies a sum to the topUsers after grouping it up by userId
tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex', 'weightedRating']]
tempTopUsersRating.columns = ['sum_similarityIndex', 'sum_weightedRating']

# Creates an empty dataframe
recommendation_df = pd.DataFrame()
# Now we take the weighted average
recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating'] / tempTopUsersRating['sum_similarityIndex']
recommendation_df['movieId'] = tempTopUsersRating.index

recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)

print(movies_df.loc[movies_df['movieId'].isin(recommendation_df.head(10)['movieId'].tolist())])
