from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, udf, lit
from pyspark.ml.feature import StringIndexer

# ساخت یک نمونه از SparkSession
spark = SparkSession.builder.getOrCreate()

# تعریف تابع بارگیری داده‌ها
def load_data(file_path):
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    return df

# تابع پیش‌پردازش داده‌ها
def preprocess_data(rating_df, movie_df):
    valid_movie_ids = rating_df.groupBy("movieId").count().filter(col("count") > 10).select("movieId")
    rating_df = rating_df.join(valid_movie_ids, on="movieId")

    valid_user_ids = rating_df.groupBy("userId").count().filter(col("count") > 10).select("userId")
    rating_df = rating_df.join(valid_user_ids, on="userId")

    rating_matrix = rating_df.groupBy("userId").pivot("movieId").agg(avg("rating")).na.fill(0)

    return rating_matrix, movie_df

# تابع یافتن n فیلم برتر برای یک کاربر
def get_top_n_recommended(user, rating_matrix, movies_df, K=10, N=5):
    user_ratings = rating_matrix.filter(col("userId") == user).first()
    if user_ratings is None:
        print(f"کاربر {user} در داده‌های رتبه‌بندی وجود ندارد.")
        return []

    nonzero_ratings = [col_name for col_name in user_ratings.asDict() if user_ratings[col_name] > 0 and col_name != 'userId']
    movies_already_seen = [int(col_name) for col_name in nonzero_ratings]

    knn_ratings = rating_matrix.filter(col("userId").isin(find_k_nearest_neighbors(user, rating_matrix, calculate_distance, K=K))).drop("userId")
    avg_rating = knn_ratings.agg(*[(avg(col(col_name)).alias(col_name)) for col_name in movies_already_seen]).first().asDict()
    top_n_movies = sorted(avg_rating.items(), key=lambda x: x[1], reverse=True)[:N]
    recommended_movies = [row.title for row in movies_df.filter(col("movieId").isin([int(movie[0]) for movie in top_n_movies])).collect()]
    return recommended_movies


# تابع یافتن k همسایه نزدیک
def find_k_nearest_neighbors(target, rating_matrix, distance_func, K=10):
    target_ratings = rating_matrix.filter(col("userId") == target).first().asDict()
    distance_udf = udf(distance_func)
    distance_df = rating_matrix.withColumn("distance", distance_udf(col("userId"), lit(target)))
    nearest_neighbors = distance_df.filter(col("userId") != target).sort("distance").limit(K)
    return nearest_neighbors.select("userId").rdd.flatMap(lambda x: x).collect()

# تابع محاسبه فاصله
def calculate_distance(user1, user2):
    common_movies = [col_name for col_name in user1.asDict() if user1[col_name] > 0 and user2[col_name] > 0]
    if len(common_movies) == 0:
        return -1
    user1_ratings = [user1[col_name] for col_name in common_movies]
    user2_ratings = [user2[col_name] for col_name in common_movies]
    return np.linalg.norm(user1_ratings - user2_ratings)

# تابع بارگیری و پیش‌پردازش داده‌ها
def load_and_preprocess_data(rating_file, movie_file):
    rating = load_data(rating_file)
    movies = load_data(movie_file)
    rating_matrix, movies = preprocess_data(rating, movies)
    return rating_matrix, movies

# ساخت DataFrame‌ها با استفاده از تابع بارگیری و پیش‌پردازش
RatingMatrix, movies = load_and_preprocess_data("../data/ratings.csv", "../data/movies.csv")

# پیشنهاد فیلم برای همه کاربران
N = 5
for user_row in RatingMatrix.collect():
    user = user_row.userId
    recommended_movies = get_top_n_recommended(user, RatingMatrix, movies, N=N)
    print(f"کاربر {user}:")
    for movie in recommended_movies:
        print(movie)
    print()
