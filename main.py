from src.Plotting.plots import *
from src.Cur.cur_decompositions import normal_cur, relerror_cur
from Madhuri_implementation.Input_pertubation import *
from sklearn import datasets
import pandas as pd
import numpy as np
from Input_pertubation.Input_pertubation import evaluate_item_averages, evaluate_user_averages, differentially_private_input_perturbation
def calculate_baselines(R):
    """
    Calculate global, item, and user averages, and adjust item averages for user biases.

    Parameters:
        R (numpy.ndarray): Rating matrix.

    Returns:
        tuple: Tuple containing global average, item averages, user averages, and item averages adjusted for user biases.
    """
    # Global average
    global_avg = np.mean(R)

    # Item averages
    item_avg_values = np.mean(R, axis=0)

    # User averages
    user_avg_values = np.mean(R, axis=1)

    # Calculate item average considering both user and item averages
    item_avg_user_adjusted = item_avg_values + np.mean(user_avg_values)

    return global_avg, item_avg_values, user_avg_values, item_avg_user_adjusted

def calculate_global_effects_for_matrix(R, item_avgs, user_avgs):
    """
    Calculate global effects for each rating in the rating matrix.

    Parameters:
        R (numpy.ndarray): The rating matrix.
        item_avgs (numpy.ndarray): Array containing the average rating for each item.
        user_avgs (numpy.ndarray): Array containing the average rating given by each user.

    Returns:
        numpy.ndarray: The matrix of global effects corresponding to each rating in the input matrix R.
    """
    global_effects = np.zeros_like(R, dtype=float)  # Initialize an empty matrix for global effects

    # Iterate through each rating in the matrix
    for i in range(R.shape[0]):  # Loop through each row (user)
        for j in range(R.shape[1]):  # Loop through each column (item)
            global_effects[i, j] = item_avgs[j] + user_avgs[i]

    return global_effects

def calculate_rmse(actual_ratings, predicted_ratings):
    """
    Calculate the Root Mean Squared Error (RMSE) between actual and predicted ratings.

    Parameters:
        actual_ratings (numpy.ndarray): Array of actual ratings.
        predicted_ratings (numpy.ndarray): Array of predicted ratings.

    Returns:
        float: The RMSE value.
    """
    mse = np.mean((actual_ratings - predicted_ratings) ** 2)
    rmse = np.sqrt(mse)
    return rmse
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
    



    # baseline calculation:

    global_avg, item_avg_values, user_avg_values, item_avg_user_adjusted = calculate_baselines(R)
    global_effects = calculate_global_effects_for_matrix(R, item_avg_values, user_avg_values)

    # Create a matrix of predicted ratings using item average baseline
    predicted_ratings_item_avg = np.tile(item_avg_values, (R.shape[0], 1))

    # Calculate RMSE for each baseline
    rmse_global_avg = calculate_rmse(R, global_avg)
    rmse_item_avg = calculate_rmse(R, predicted_ratings_item_avg)
    rmse_global_effects = calculate_rmse(R, global_effects)



    #functions
    #for plotting rmse vs no. of features selected 
    
    epsilon=0.35
    global_avg, item_avgs = evaluate_item_averages(R, beta_i, 0.14*epsilon, 0.14*epsilon, r_min, r_max,delta_r)
    user_avgs = evaluate_user_averages(R, item_avgs, beta_i, 0.14*epsilon, 0.14*epsilon, u_min, u_max,delta_r)
    private_R= differentially_private_input_perturbation(R,beta_i, 0.7*epsilon,delta_r)
    
    private_R=pd.DataFrame(private_R)
    
    #madhuri_implementation
    global_avg, item_avgs = evaluate_item_averages_madhuri(R, beta_i, 0.14*epsilon, 0.14*epsilon, r_min, r_max,delta_r)
    user_avgs = evaluate_user_averages_madhuri(R, item_avgs, beta_i, 0.14*epsilon, 0.14*epsilon, u_min, u_max,delta_r)
    madhuri_R= differentially_private_input_perturbation_madhuri(R,beta_i, 0.7*epsilon,delta_r)
    
    madhuri_R=pd.DataFrame(madhuri_R)
    #for rmse vs feature selected plot
    R=pd.DataFrame(R)
    plot_curs_dist_dual(R,private_R,madhuri_R, 10, 13, 25, "./Figs/")
    
    #starting with epsilon comparison
    '''epsilon = [i / 10 for i in range(1, 6)]

    orig_subspace_cur_dist = []
    orig_leverage_cur_dist = []
    pert_subspace_cur_dist = []
    pert_leverage_cur_dist = []
    pert_subspace_cur_dist_madhuri = []
    pert_leverage_cur_dist_madhuri = []

    for i in epsilon:
        global_avg, item_avgs = evaluate_item_averages(R, beta_i, 0.14 * i, 0.14 * i, r_min, r_max, delta_r)
        user_avgs = evaluate_user_averages(R, item_avgs, beta_i, 0.14 * i, 0.14 * i, u_min, u_max, delta_r)
        private_R = differentially_private_input_perturbation(R, beta_i, 0.7 * i, delta_r)
        private_R = pd.DataFrame(private_R)

        # Preprocess madhuri_df
        global_avg_madhuri, item_avgs_madhuri = evaluate_item_averages_madhuri(R, beta_i, 0.14 * i, 0.14 * i, r_min, r_max, delta_r)
        user_avgs_madhuri = evaluate_user_averages_madhuri(R, item_avgs_madhuri, beta_i, 0.14 * i, 0.14 * i, u_min, u_max, delta_r)
        private_R_madhuri = differentially_private_input_perturbation_madhuri(R, beta_i, 0.7 * i, delta_r)
        private_R_madhuri = pd.DataFrame(private_R_madhuri)

        R_generated = pd.DataFrame(R)  # always getting the original matrix for RMSE calculation
        orig_subspace_cur_dist1, orig_leverage_cur_dist1, pert_subspace_cur_dist1, pert_leverage_cur_dist1, pert_subspace_cur_dist1_madhuri, pert_leverage_cur_dist1_madhuri = plot_curs_dist_dual_epsilon(R_generated, private_R, private_R_madhuri, 100, 13, 25, "./Figs2/epsilon_comparison/")
        orig_subspace_cur_dist.extend(orig_subspace_cur_dist1)
        orig_leverage_cur_dist.extend(orig_leverage_cur_dist1)
        pert_subspace_cur_dist.extend(pert_subspace_cur_dist1)
        pert_leverage_cur_dist.extend(pert_leverage_cur_dist1)
        pert_subspace_cur_dist_madhuri.extend(pert_subspace_cur_dist1_madhuri)
        pert_leverage_cur_dist_madhuri.extend(pert_leverage_cur_dist1_madhuri)

    global_avg_values = [rmse_global_avg for i in range(len(epsilon))]
    item_avg_values = [rmse_item_avg for i in range(len(epsilon))] 
    global_effect_values=[rmse_global_effects for  i in range(len(epsilon))]
# Adding labels and title
    # Determine the maximum and minimum values among all data points
    

    # Adding labels and title
    # Determine the maximum and minimum values among all data points
    all_data = [orig_subspace_cur_dist, orig_leverage_cur_dist, pert_subspace_cur_dist, pert_leverage_cur_dist,
                global_avg_values, item_avg_values, global_effect_values, pert_subspace_cur_dist_madhuri,
                pert_leverage_cur_dist_madhuri]
    min_val = min(min(data) for data in all_data)
    max_val = max(max(data) for data in all_data)

    # Increase figure size
    plt.figure(figsize=(10, 6))  # Adjust width and height as needed

    # Plotting
    plt.plot(epsilon, orig_subspace_cur_dist, label='Original RELCUR/CUR', linestyle='-', marker='o', color='blue')

    plt.plot(epsilon, pert_subspace_cur_dist, label='Perturbed RELCUR', linestyle='-.', marker='s', color='green')
    plt.plot(epsilon, pert_leverage_cur_dist, label='Perturbed CUR', linestyle=':', marker='^', color='red')

    # Plotting additional curves
    plt.plot(epsilon, pert_subspace_cur_dist_madhuri, label='Perturbed RELCUR(Madhuri)', linestyle='-.', marker='*', color='cyan')
    plt.plot(epsilon, pert_leverage_cur_dist_madhuri, label='Perturbed CUR(Madhuri)', linestyle=':', marker='x', color='magenta')
    plt.plot(epsilon, global_avg_values, label='Global Average', linestyle='-', marker='.', color='black')
    plt.plot(epsilon, item_avg_values, label='Item Average', linestyle='--', marker='>', color='purple')
    plt.plot(epsilon, global_effect_values, label='Global Effect', linestyle='-.', marker='p', color='orange')

    # Set y-axis limits
    plt.ylim(min_val-0.1, max_val+0.2)

    # Set y-axis ticks with interval of 0.05
    plt.yticks(np.arange(min_val-0.1, max_val+0.2, 0.05))

    plt.xlabel('Privacy parameter epsilon')
    plt.ylabel('RMSE')
    plt.title('Epsilon Comparison')
    plt.legend(loc='upper right')  # Adjust the legend location as needed

    # Saving the plot
    plt.savefig('epsilon_comparison_madhuri.png')
    plt.show()
'''