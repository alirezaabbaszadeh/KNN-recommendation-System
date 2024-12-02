import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import timeit
# بهینه‌سازی عملیات خواندن داده‌ها از فایل
def load_data(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    return df
# بهینه‌سازی عملیات پیش‌پردازش داده‌ها
def preprocess_data(rating_df, movie_df):
    valid_movie_ids = rating_df['movieId'].value_counts()[rating_df['movieId'].value_counts() > 10].index
    rating_df = rating_df[rating_df['movieId'].isin(valid_movie_ids)]
    valid_user_ids = rating_df['userId'].value_counts()[rating_df['userId'].value_counts() > 10].index
    rating_df = rating_df[rating_df['userId'].isin(valid_user_ids)]
    rating_matrix = pd.pivot_table(rating_df, values='rating', index='userId', columns='movieId').fillna(0)
    return rating_matrix, movie_df
# بهینه‌سازی عملیات پیشنهاد n فیلم برتر
def get_top_n_recommended(user, rating_matrix, movies_df, K=10, N=5):

    neighbors = find_k_nearest_neighbors(user, rating_matrix, calculate_distance, K)

    neighbor_indices = [neighbor[0] for neighbor in neighbors]
    knn_ratings = rating_matrix.iloc[neighbor_indices]
    movies_already_seen = rating_matrix.loc[user].values.nonzero()[0]
    avg_rating = knn_ratings.mean(axis=0).iloc[movies_already_seen].dropna()
    top_n_movies = avg_rating.nlargest(N).index
    recommended_movies = movies_df[movies_df['movieId'].isin(top_n_movies)]['title'].tolist()
    return recommended_movies
# بهینه‌سازی عملیات یافتن k همسایه نزدیک
def find_k_nearest_neighbors(target, rating_matrix, distance_func, K=10):
    target_rating = rating_matrix.loc[target]
    distances = cdist([target_rating], rating_matrix, metric=distance_func)[0]
    nearest_neighbors_indices = np.argpartition(distances, K + 1)[:K + 1]
    nearest_neighbors = [(user, distances[user]) for user in nearest_neighbors_indices if user != target]
    nearest_neighbors.sort(key=lambda x: x[1])
    return nearest_neighbors[:K]
# بهینه‌سازی عملیات محاسبه فاصله
def calculate_distance(user1rating, user2rating):
    common_movies = np.logical_and(user1rating > 0, user2rating > 0)
    if np.sum(common_movies) == 0:
        return -1
    user1 = user1rating[common_movies]
    user2 = user2rating[common_movies]
    return np.linalg.norm(user1 - user2)
# بهینه‌سازی عملیات خواندن داده‌ها و پیش‌پردازش
def load_and_preprocess_data(rating_file, movie_file):
    rating = load_data(rating_file)
    movies = load_data(movie_file)
    rating_matrix, movies = preprocess_data(rating, movies)
    return rating_matrix, movies
# تست زمان اجرا
start_time = timeit.default_timer()
# بهینه‌سازی عملیات خواندن داده‌ها و پیش‌پردازش
RatingMatrix, movies = load_and_preprocess_data("D:/uni/5/KNN/cd/recom-sys/data/ratings.csv", "D:/uni/5/KNN/cd/recom-sys/data/movies.csv")
# پیشنهاد فیلم برای همه کاربران
N = 5
for user in RatingMatrix.index:
    recommended_movies = get_top_n_recommended(user, RatingMatrix, movies, N=N)
    print(f"USER {user}:")
    for movie in recommended_movies:
        print(movie)
    print()
# محاسبه زمان اجرا
elapsed = timeit.default_timer() - start_time
print("زمان اجرا:", elapsed, "ثانیه")
