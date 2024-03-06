from src.Plotting.plots import plot_curs_dist, plot_curs_time, plot_cur_runtime
from src.Cur.cur_decompositions import leverage_cur, subspace_cur
from sklearn import datasets
import pandas as pd
import numpy as np
from Input_pertubation.Input_pertubation import evaluate_item_averages, evaluate_user_averages, differentially_private_input_perturbation

def get_user_movie_rating_matrix(file_path):
    """
    Loads data from the specified file and constructs the user_movie_rating_matrix

    Args:
        file_path (str): Path to the data file.

    Returns:
        user_movie_rating_matrix (np.ndarray): User-movie rating matrix.
    """
    df = pd.read_csv(file_path, sep="\t", names=['user_id', 'movie_id', 'rating', 'timestamp'])
    num_users = df['user_id'].max()
    num_movies = df['movie_id'].max()

    user_movie_rating_matrix = np.zeros((num_users, num_movies))

    for i in range(len(df)):
        user_id = int(df['user_id'][i]) - 1
        movie_id = int(df['movie_id'][i]) - 1
        rating = float(df['rating'][i])

        user_movie_rating_matrix[user_id][movie_id] = rating

    return pd.DataFrame(user_movie_rating_matrix)
def dataprocessing():
    file_path = 'ml-100k/u.data'  # Specify the correct file path to your data file
    user_movie_matrix = get_user_movie_rating_matrix(file_path)

    return user_movie_matrix
    #print(ratings_matrix)
    
if __name__ == '__main__':
    print("demo run")
    #boston = pd.DataFrame(load_boston(return_X_y=True)[0])
    #plot_curs_dist(boston, 10, 13, 25, "./Figs/boston/")
    #plot_curs_time(boston, 10, 13, 25, "./Figs/boston/")
    ''' boston = pd.DataFrame(load_boston(return_X_y=True)[0])
    boston = boston - boston.mean(axis=0)'''
    #plot_cur_runtime(boston, 4, 10, 20, 15, "./Figs/boston/", data_name="Boston")
    '''print("loading wine")
    wine = pd.DataFrame(datasets.load_wine(return_X_y=True)[0])
    print(wine)'''
    #wine = wine - wine.mean(axis=0)
    #plot_curs_dist(wine, 10, 13, 25, "./Figs/wine1/")
    ratings_matrix_filled=dataprocessing()
    '''print(ratings_matrix_filled)
    plot_curs_dist(ratings_matrix_filled, 10, 13, 25, "./Figs/ml-1m/")'''
    '''# Centering the data
    #ratings_matrix_centered = ratings_matrix_filled - ratings_matrix_filled.mean(axis=0)
    plot_curs_dist(ratings_matrix_filled, 10, 13, 25, "./Figs/ml-1m/")
   
'''
    #for input pertubation
    #parameters
    R = np.array(ratings_matrix_filled)
    delta_r = ratings_matrix_filled.values.max() - ratings_matrix_filled.values.max()
    r_min = ratings_matrix_filled.values.min()
    r_max =ratings_matrix_filled.values.max()
    u_min = -2
    u_max = 2
    beta_i = (ratings_matrix_filled.values.max() + ratings_matrix_filled.values.min())//2

    #functions 
    global_avg, item_avgs = evaluate_item_averages(R, beta_i, 0.14*2, 0.14*2, r_min, r_max,delta_r)
    user_avgs = evaluate_user_averages(R, item_avgs, beta_i, 0.14*2, 0.14*2, u_min, u_max,delta_r)
    private_R= differentially_private_input_perturbation(R,beta_i, 0.7*2,delta_r)

    private_R=pd.DataFrame(private_R)
 
    plot_curs_dist(private_R, 10, 13, 25, "./Figs2/input_pertubation1/")
    
   #suidgcid