import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import timeit

# بهینه‌سازی عملیات خواندن داده‌ها از فایل

def load_data(file_path):

    # خواندن داده‌ها از فایل به کمک pandas و ذخیره در دیتافریم df

    df = pd.read_csv(file_path, low_memory=False)

    return df

# بهینه‌سازی عملیات پیش‌پردازش داده‌ها

def preprocess_data(rating_df, movie_df):

    # دریافت شناسه‌های فیلم‌های معتبر که تعداد رتبه‌بندی‌های بیشتر از ۱۰ دارند

    valid_movie_ids = rating_df['movieId'].value_counts()[rating_df['movieId'].value_counts() > 10].index

    # فیلتر کردن رتبه‌بندی‌ها بر اساس فیلم‌های معتبر

    rating_df = rating_df[rating_df['movieId'].isin(valid_movie_ids)]

    # دریافت شناسه‌های کاربران معتبر که تعداد رتبه‌بندی‌های بیشتر از ۱۰ دارند

    valid_user_ids = rating_df['userId'].value_counts()[rating_df['userId'].value_counts() > 10].index

    # فیلتر کردن رتبه‌بندی‌ها بر اساس کاربران معتبر

    rating_df = rating_df[rating_df['userId'].isin(valid_user_ids)]

    # ساختن ماتریس رتبه‌بندی بر اساس رتبه‌بندی‌های موجود و پرکردن خانه‌های خالی با صفر

    rating_matrix = pd.pivot_table(rating_df, values='rating', index='userId', columns='movieId').fillna(0)

    return rating_matrix, movie_df




def get_top_n_recommended(user, rating_matrix, movies_df, K=10, N=5):
    """
    برای کاربر داده شده، لیستی از فیلم‌های پیشنهادی را برمی‌گرداند.

    پارامترها:
        user (str): کاربری که پیشنهادها برای او ایجاد می‌شوند.
        rating_matrix (pandas.DataFrame): ماتریسی که شامل امتیازهای کاربران برای فیلم‌هاست.
        movies_df (pandas.DataFrame): DataFrame شامل اطلاعات فیلم‌ها.
        K (int، اختیاری): تعداد همسایگان نزدیک که در نظر گرفته می‌شود. مقدار پیش‌فرض ۱۰ است.
        N (int، اختیاری): تعداد فیلم‌های برتر برای پیشنهاد. مقدار پیش‌فرض ۵ است.

    برگرداندن:
        list: لیستی از فیلم‌های پیشنهادی.

    """

    # یافتن K همسایه نزدیک

    neighbors = find_k_nearest_neighbors(user, rating_matrix, calculate_distance, K)

    # استخراج شاخص‌های همسایگان

    neighbor_indices = [neighbor[0] for neighbor in neighbors]

    # دریافت امتیازهای همسایگان نزدیک

    knn_ratings = rating_matrix.iloc[neighbor_indices]

    # دریافت شاخص‌های فیلم‌هایی که کاربر قبلاً دیده است

    movies_already_seen = rating_matrix.loc[user].values.nonzero()[0]

    # محاسبه میانگین امتیاز فیلم‌های دیده شده توسط همسایگان نزدیک

    avg_rating = knn_ratings.mean(axis=0).iloc[movies_already_seen].dropna()

    # دریافت N فیلم برتر براساس میانگین امتیاز

    top_n_movies = avg_rating.nlargest(N).index

    # دریافت عناوین فیلم‌های پیشنهادی

    recommended_movies = movies_df[movies_df['movieId'].isin(top_n_movies)]['title'].tolist()

    return recommended_movies



def find_k_nearest_neighbors(target, rating_matrix, distance_func, K=10):
    """
    برای کاربر هدف، K نزدیک‌ترین همسایه‌ها را در یک ماتریس امتیازدهی بر اساس یک تابع فاصله پیدا می‌کند.

    آرگومان‌ها:
        target (int): شاخص کاربر هدف در ماتریس امتیازدهی.
        rating_matrix (pandas.DataFrame): ماتریس حاوی امتیازهای کاربران.
        distance_func (callable): تابعی برای محاسبه فاصله بین کاربران.
        K (int): تعداد نزدیک‌ترین همسایه‌ها برای پیدا کردن (پیش‌فرض 10).

    بازگرداندن:
        list: لیستی از تاپل‌هایی که شامل شاخص‌های نزدیک‌ترین همسایه‌ها و فاصله آن‌ها تا کاربر هدف است.

    """

    # بدست آوردن امتیاز کاربر هدف

    target_rating = rating_matrix.loc[target]

    # محاسبه فواصل بین کاربر هدف و سایر کاربران

    distances = cdist([target_rating], rating_matrix, metric=distance_func)[0]

    # بدست آوردن شاخص‌های K+1 نزدیک‌ترین همسایه‌ها

    nearest_neighbors_indices = np.argpartition(distances, K + 1)[:K + 1]

    # ایجاد لیستی از همسایه‌های نزدیک، بدون شامل کاربر هدف

    nearest_neighbors = [(user, distances[user]) for user in nearest_neighbors_indices if user != target]

    # مرتب‌سازی همسایه‌های نزدیک بر اساس فاصله آن‌ها

    nearest_neighbors.sort(key=lambda x: x[1])

    # بازگرداندن K نزدیک‌ترین همسایه‌ها

    return nearest_neighbors[:K]







def calculate_distance(user1rating, user2rating):

    # پیدا کردن فیلم‌های مشترک بین دو کاربر

    common_movies = np.logical_and(user1rating > 0, user2rating > 0)

    # اگر تعداد فیلم‌های مشترک صفر باشد، به عنوان نتیجه -1 را برگردان

    if np.sum(common_movies) == 0:

        return -1

    # استخراج امتیازات کاربران برای فیلم‌های مشترک

    user1 = user1rating[common_movies]

    user2 = user2rating[common_movies]

    # محاسبه فاصله بین دو کاربر با استفاده از نرم اقلیدسی

    return np.linalg.norm(user1 - user2)







def load_and_preprocess_data(rating_file, movie_file):

    # این تابع داده‌های امتیازدهی را از فایل مشخصی بارگیری می‌کند

    rating = load_data(rating_file)

    # این تابع داده‌های فیلم‌ها را از فایل مشخصی بارگیری می‌کند

    movies = load_data(movie_file)

    # این تابع داده‌های امتیازدهی و فیلم‌ها را پیش‌پردازش می‌کند و ماتریس امتیازدهی و فیلم‌ها را برمی‌گرداند

    rating_matrix, movies = preprocess_data(rating, movies)

    # ماتریس امتیازدهی و فیلم‌های پیش‌پردازش شده را برمی‌گرداند

    return rating_matrix, movies


# تست زمان اجرا

start_time = timeit.default_timer()

# بهینه‌سازی عملیات خواندن داده‌ها و پیش‌پردازش

RatingMatrix, movies = load_and_preprocess_data("../data/ratings.csv", "../data/movies.csv")

# پیشنهاد فیلم برای همه کاربران

N = 5

for user in RatingMatrix.index:

    recommended_movies = get_top_n_recommended(user, RatingMatrix, movies, N=N)

    print(f"کاربر {user}:")

    for movie in recommended_movies:

        print(movie)

    print()



# محاسبه زمان اجرا

elapsed = timeit.default_timer() - start_time

print("زمان اجرا:", elapsed, "ثانیه")
