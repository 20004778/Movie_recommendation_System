# Movie_recommendation_System
## Algorithm:
In this System is give the recommendation based on review ratings.
## Collaborative filtering
The idea of a collaborative filtering approach is to collect and analyze a large amount of information about user actions and settings and then predict which users will favor their similarity with other users. The advantage of collaborative filtering is that it does not rely on content that can be analyzed and can accurately represent complex items. Algorithms are used to calculate user similarities or item similarities in recommender systems, such as k-nearest neighbors and Pearson correlation. Another collaborative filtering concept is based on assumptions. People who buy in the past will buy in the future and like the same product them like in the past.
```python
import numpy as np
import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from ipywidgets import interact
data = pd.read_csv('/content/movies.csv')
data.head()
data.info()
data.shape
data.describe()
rating = pd.read_csv('/content/ratings.csv')
rating.head()
rating.info()
rating.describe()
rating.shape
data = pd.merge(data, rating, on = 'movieId', how = 'inner')
data.head()
data.shape
data.info()
data = data.drop(['movieId', 'userId', 'timestamp'], axis = 1)
data.head()
data = pd.pivot_table(data, index = ['title','genres'], aggfunc = 'mean')
data.reset_index(level=['title','genres'], inplace = True)
data.head()
data['y'] = data['title'].str.split(' ')
data['year'] = data['y'].apply(lambda x: x[-1])
data.head()
data = data.drop(['y'], axis = 1)
data.head()
data['year'] = data['year'].str.strip(')')
data['year'] = data['year'].str.strip('(')
data.head()
data['year'].value_counts()[:5]
data['year'].unique()
data[data.year.isin(['Road', ''])]
data['year'] = data['year'].replace(('Road', ''),                                   ('2015','2011'))
data['year'].unique()
data['year'] = data['year'].astype(int)
data.info()
data.head()
data['title'] = data['title'].str.split(' ')
data['title'] = data['title'].apply(lambda x: ' '.join(x[:-1]))
data.head()
data[data['year'] == data['year'].max()][['title','rating']].sort_values(by = 'rating',
                ascending = False).head(10).reset_index(drop = True).style.background_gradient(cmap = 'Wistia')
print("The Number of Movies that received 5 Star Reviews :", data[data['rating'] == 5]['title'].count())
print("Percentage of Movies Getting 5 Star Reviews : {0:.2f}%".format((data[data['rating'] == 5]['title'].count())/
                                                                      (data.shape[0])))
print("\nThe Number of Movies that received less than 1 Star Reviews :", data[data['rating'] <= 1]['title'].count())
print("The Percentage of Movies Getting Less than 1 Star Reviews : {0:.2f}%".format((
    data[data['rating'] <= 1]['title'].count())/(data.shape[0])))
@interact
def genre(Genre = ['Action', 'Adventure', 'Animation','Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
       'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance','Sci-Fi', 'Thriller', 'War', 'Western'],
          year = sorted(data['year'].unique(), reverse=True)):
    x = data['genres'].str.split('|')
    d = data.drop(['genres'], axis = 1)
    x = pd.concat([d, x], axis = 1)
    x = x.explode('genres')
    x= x[(x['genres'] == Genre)& 
         (x['year'] >= year)][['title', 'rating', 'year']].sort_values(by = ['rating','year'],
                            ascending = [False,True]).reset_index(drop = True).head(10)
    return x
from mlxtend.preprocessing import TransactionEncoder
genres = data['genres'].str.split('|')
te = TransactionEncoder()
genres = te.fit_transform(genres)
genres = pd.DataFrame(genres, columns = te.columns_)
genres.head()
genres = genres.astype('int')
genres.insert(0, 'title', data['title'])
genres.head()
genres.rename(columns = genres.iloc[0])
genres = genres.transpose()
genres = genres.rename(columns = genres.iloc[0])
genres = genres.drop(genres.index[0])
genres = genres.astype(int)
genres.head()
sorted(data['year'].unique(), reverse=True)
@interact
def recommendation_movie(movie = sorted(genres.columns.unique(), reverse=False)):    
    similar_movies = genres.corrwith(genres[movie])
    similar_movies = similar_movies.sort_values(ascending=False)
    similar_movies=similar_movies.dropna()
    similar_movies=similar_movies.reset_index()
    similar_movies.columns = ['Movie', 'Score']
    similar_movies=similar_movies[similar_movies['Movie']!=movie]
    similar_movies.Score=round((similar_movies.Score*100),2)
    if len(similar_movies)== 0:
        return print('\n\t\t No Recommendation!!!')
    else:
        return similar_movies.head()
```
### OUTPUT:
#### movie dataset
![image](https://user-images.githubusercontent.com/75236145/232328241-76755258-f5d6-4e0d-8f85-bc882180e9aa.png)
#### rating dataset
![image](https://user-images.githubusercontent.com/75236145/232328572-f32dd266-bd51-4161-8eb4-d9e54a30f018.png)
![image](https://user-images.githubusercontent.com/75236145/232328623-33bbb628-38fd-4cf0-bbe3-126bca73bc83.png)

![image](https://user-images.githubusercontent.com/75236145/232328646-7ceb774f-8b5e-403c-a8d9-c176dc4d0a26.png)


