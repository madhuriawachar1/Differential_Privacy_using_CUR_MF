from cur_decomposition import get_user_movie_rating_matrix

def dataprocessing():
    file_path = 'ml-100k/u.data'  # Specify the correct file path to your data file
    user_movie_matrix = get_user_movie_rating_matrix(file_path)

    return user_movie_matrix


if __name__ == '__main__':
    print("demo run")
    
    ratings_matrix_filled=dataprocessing()
    
    