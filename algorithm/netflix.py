import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean





#step 1 ...

# خواندن داده‌های رتبه‌بندی از فایل CSV

rating = pd.read_csv("D:/uni/5/KNN/cd/recom-sys/data/ratings.csv", usecols=[0, 1, 2])


# خواندن داده‌های فیلم‌ها از فایل CSV

movies = pd.read_csv("D:/uni/5/KNN/cd/recom-sys/data/movies.csv", usecols=[0, 1])



# تعداد کاربران برای هر فیلم

userPerMovies = rating.movieId.value_counts()


# حذف رتبه‌بندی‌هایی که تعداد کاربران آنها کمتر از ۱۰ است

rating = rating[rating["movieId"].isin(userPerMovies[userPerMovies > 10].index)]


# تعداد فیلم‌ها برای هر کاربر

MoviePerUser = rating.userId.value_counts()


# حذف کاربرانی که تعداد فیلم‌های آنها کمتر از ۱۰ است

rating = rating[rating["userId"].isin(MoviePerUser[MoviePerUser > 10].index)]


# ساختن ماتریس رتبه‌بندی

RatingMatrix = pd.pivot_table(rating, values="rating", index=["userId"], columns=["movieId"])







#step2 ...

#  این تابع برای محاسبه فاصله بین دو کاربر استفاده می‌شود.
# تابع محاسبه فاصله


def distance(user1ID, user2ID):


    user1rating = RatingMatrix.transpose()[user1ID]

    user2rating = RatingMatrix.transpose()[user2ID]



    # فیلم‌هایی که دو کاربر مشترک دارند

    common_movies = []

    for column in RatingMatrix.columns:

        if user1rating[column] > 0 and user2rating[column] > 0:


            common_movies.append(column)

    # نگه‌داشتن رتبه‌بندی‌های موجود

    user1 = []

    user2 = []




    for i in common_movies:

        user1.append(user1rating[i])

        user2.append(user2rating[i])



    # محاسبه فاصله اقلیدسی

    if len(common_movies) == 0:

        distance = -1



    else:

        distance = euclidean(user1, user2)

    return distance




#setep 3 ...

# تابع یافتن k همسایه نزدیک
def knearestNeighbors(target, K=10):


    common_users = []

    distance_to_target = []


    # افزودن فاصله برای هر کاربر مشترک

    for user in RatingMatrix.index:


        if distance(target, user) != -1:


            common_users.append(user)

            distance_to_target.append(distance(target, user))


    userDataFrame = pd.DataFrame(distance_to_target, common_users, columns=["distance"])


    # یافتن k همسایه نزدیک به کاربر هدف

    KNearestNeighbors = userDataFrame.sort_values(["distance"], ascending=True)[:K]


    return KNearestNeighbors






# step 4 ...

# تابع پیشنهاد n فیلم برتر


def TopNRecommended(user, N=5):

    a = knearestNeighbors(user)




    # رتبه‌بندی‌های k نزدیک‌ترین همسایه

    KNNRatings = RatingMatrix[RatingMatrix.index.isin(a.index)]


    # محاسبه میانگین رتبه‌بندی‌ها

    avgRating = KNNRatings.apply(np.mean).dropna()


    # فیلم‌هایی که کاربر هدف قبلاً دیده است

    MoviesAlreadySeen = RatingMatrix.transpose()[user].dropna().index

    avgRating = avgRating[~avgRating.isin(MoviesAlreadySeen)]


    # یافتن n فیلم برتر برای کاربر هدف

    TopNmovies = avgRating.sort_values(ascending=False).index[:N]

    # پیشنهاد n فیلم برتر به کاربر هدف

    RecommendedMovies = []

    for movie in TopNmovies:

        movie_title = movies.loc[movies['movieId'] == movie, 'title'].values[0]

        RecommendedMovies.append(movie_title)

    return RecommendedMovies
