Here’s a complete `README.md` file for your repository:

```markdown
# Movie Recommendation System using PySpark

This repository implements a collaborative filtering-based movie recommendation system using **PySpark**. The system processes user movie ratings data and generates personalized movie recommendations based on user preferences and the preferences of similar users.

The recommendation engine uses a **k-nearest neighbors (KNN)** approach to find users with similar movie preferences and recommends movies they have rated highly. The project demonstrates how to work with large datasets using PySpark for efficient computation and distributed data processing.

## Key Features:
- **Data Preprocessing**: Cleans and filters the dataset by removing users and movies with fewer than 10 ratings.
- **Collaborative Filtering**: Implements k-nearest neighbors to identify similar users based on their movie ratings.
- **Movie Recommendations**: Recommends the top `N` movies for each user based on the ratings of their nearest neighbors.
- **Distributed Computing**: Utilizes PySpark for distributed data processing, allowing the handling of large datasets.
  
## Requirements:
- **PySpark**: A powerful framework for distributed computing.
- **Numpy**: Used for numerical calculations, especially in distance computation.

You can install the required dependencies using pip:

```bash
pip install pyspark numpy
```

## Files:
1. **`ratings.csv`**: Contains user movie ratings with columns `userId`, `movieId`, and `rating`.
2. **`movies.csv`**: Contains movie information with columns `movieId` and `title`.

## Installation & Setup

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/your-username/movie-recommendation-pyspark.git
    cd movie-recommendation-pyspark
    ```

2. Download or prepare your own **ratings.csv** and **movies.csv** files. You can find sample datasets like [MovieLens](https://grouplens.org/datasets/movielens/) to test the system.

3. Place the CSV files in the **`data/`** directory.

4. Run the code using PySpark:

    ```bash
    python movie_recommendation.py
    ```

## Functions:
### 1. `load_data(file_path)`
Loads the dataset from a CSV file into a PySpark DataFrame.

#### Parameters:
- `file_path`: Path to the CSV file.

#### Returns:
- PySpark DataFrame containing the data.

### 2. `preprocess_data(rating_df, movie_df)`
Filters and preprocesses the data, removing movies and users with fewer than 10 ratings.

#### Parameters:
- `rating_df`: DataFrame containing user ratings.
- `movie_df`: DataFrame containing movie information.

#### Returns:
- A tuple: 
  - Preprocessed `rating_matrix` with movies and users filtered.
  - `movie_df`.

### 3. `get_top_n_recommended(user, rating_matrix, movies_df, K=10, N=5)`
Recommends the top `N` movies for a given user based on ratings from their `K` nearest neighbors.

#### Parameters:
- `user`: User ID for whom the recommendations are to be generated.
- `rating_matrix`: DataFrame containing ratings of all users.
- `movies_df`: DataFrame containing movie details.
- `K`: Number of nearest neighbors (default is 10).
- `N`: Number of recommended movies (default is 5).

#### Returns:
- A list of the top `N` recommended movies.

### 4. `find_k_nearest_neighbors(target, rating_matrix, distance_func, K=10)`
Identifies the `K` nearest neighbors for a given user using the specified distance function.

#### Parameters:
- `target`: User ID of the target user.
- `rating_matrix`: DataFrame containing ratings of all users.
- `distance_func`: Function used to compute the distance between users.
- `K`: Number of nearest neighbors (default is 10).

#### Returns:
- A list of `K` nearest user IDs.

### 5. `calculate_distance(user1, user2)`
Computes the distance between two users based on their movie ratings.

#### Parameters:
- `user1`: Ratings of the first user.
- `user2`: Ratings of the second user.

#### Returns:
- The distance between the two users.

## Example Usage:
```python
# Load and preprocess data
RatingMatrix, movies = load_and_preprocess_data("../data/ratings.csv", "../data/movies.csv")

# Recommend top 5 movies for each user
N = 5
for user_row in RatingMatrix.collect():
    user = user_row.userId
    recommended_movies = get_top_n_recommended(user, RatingMatrix, movies, N=N)
    print(f"کاربر {user}:")
    for movie in recommended_movies:
        print(movie)
    print()
```

This code will print the top 5 recommended movies for each user based on the collaborative filtering algorithm.

## Code Overview:
### 1. **Spark Session Creation:**
The code starts by creating a Spark session using `SparkSession.builder.getOrCreate()`, which enables distributed processing with PySpark.

### 2. **Data Loading:**
The function `load_data(file_path)` loads CSV files into PySpark DataFrames.

### 3. **Data Preprocessing:**
The `preprocess_data()` function filters the rating data by selecting users and movies with more than 10 ratings, ensuring the data quality.

### 4. **Recommendation System:**
- The `get_top_n_recommended()` function generates movie recommendations based on the average ratings of a user’s nearest neighbors.
- The `find_k_nearest_neighbors()` function identifies the closest users using a distance function.
- The `calculate_distance()` function calculates the Euclidean distance between two users based on their shared movie ratings.

### 5. **Main Logic:**
The main part of the code processes data, computes recommendations for each user, and prints the results.

## License:
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### Additional Information:
- The project uses **PySpark** to efficiently handle large datasets in a distributed manner.
- It uses a simple **k-nearest neighbors** algorithm for collaborative filtering. More sophisticated techniques, like matrix factorization or deep learning, could be used for improved recommendations.
- This repository is designed to help you understand how collaborative filtering works and how to implement it using PySpark.

---

### Contributing:
Feel free to fork the repository, make changes, and open pull requests. If you have any questions, open an issue.

---

Let me know if you need any further adjustments!
