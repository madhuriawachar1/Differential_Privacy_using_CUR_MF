
from src3.plotting.plots import plot_curs_dist
from src3.Cur.rel_cur_decomposition import subspace_cur
from src3.Cur.cur_decomposition import *
from sklearn import datasets
import pandas as pd
import numpy as np
def dataprocessing():
    file_path = 'ml-100k/u.data'  # Specify the correct file path to your data file
    user_movie_matrix = get_user_movie_rating_matrix(file_path)

    return user_movie_matrix


if __name__ == '__main__':
    print("demo run")
    
    ratings_matrix_filled=dataprocessing()
    plot_curs_dist(pd.DataFrame(ratings_matrix_filled), 10, 13, 25, "./Figs3/")
    