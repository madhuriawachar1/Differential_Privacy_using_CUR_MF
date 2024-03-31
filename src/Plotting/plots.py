import numpy as np
import pandas as pd
import time
from typing import Callable

from ..Cur.cur_decompositions import *
import matplotlib.pyplot as plt
   
import numpy as np
import pandas as pd

def sample_from_leverage_cur_dual(original_df1: pd.DataFrame, original_df: pd.DataFrame, perturbed_df: pd.DataFrame, madhuri_df: pd.DataFrame, sing_vect_num: int, search_size: int, sample_size=50):
    '''Helper function to sample from the leverage CUR decomposition for original, perturbed, and madhuri datasets.'''
    # Sample from leverage CUR for original dataset
    orig_leverage_row_vals = np.empty(sample_size)
    orig_leverage_col_vals = np.empty(sample_size)
    orig_leverage_cur_vals = np.empty(sample_size)
    for i in range(sample_size):
        _, _, _, orig_dists = normal_cur(original_df1, original_df, sing_vect_num, search_size)
        orig_leverage_row_vals[i] = orig_dists["row_dist"]
        orig_leverage_col_vals[i] = orig_dists["col_dist"]
        orig_leverage_cur_vals[i] = orig_dists["cur_dist"]

    # Sample from leverage CUR for perturbed dataset
    pert_leverage_row_vals = np.empty(sample_size)
    pert_leverage_col_vals = np.empty(sample_size)
    pert_leverage_cur_vals = np.empty(sample_size)
    for i in range(sample_size):
        _, _, _, pert_dists = normal_cur(original_df1, perturbed_df, sing_vect_num, search_size)
        pert_leverage_row_vals[i] = pert_dists["row_dist"]
        pert_leverage_col_vals[i] = pert_dists["col_dist"]
        pert_leverage_cur_vals[i] = pert_dists["cur_dist"]

    # Sample from leverage CUR for madhuri dataset
    madhuri_leverage_row_vals = np.empty(sample_size)
    madhuri_leverage_col_vals = np.empty(sample_size)
    madhuri_leverage_cur_vals = np.empty(sample_size)
    for i in range(sample_size):
        _, _, _, madhuri_dists = normal_cur(original_df1, madhuri_df, sing_vect_num, search_size)
        madhuri_leverage_row_vals[i] = madhuri_dists["row_dist"]
        madhuri_leverage_col_vals[i] = madhuri_dists["col_dist"]
        madhuri_leverage_cur_vals[i] = madhuri_dists["cur_dist"]

    return (np.average(orig_leverage_row_vals), np.std(orig_leverage_row_vals),
            np.average(orig_leverage_col_vals), np.std(orig_leverage_col_vals),
            np.average(orig_leverage_cur_vals), np.std(orig_leverage_cur_vals),
            np.average(pert_leverage_row_vals), np.std(pert_leverage_row_vals),
            np.average(pert_leverage_col_vals), np.std(pert_leverage_col_vals),
            np.average(pert_leverage_cur_vals), np.std(pert_leverage_cur_vals),
            np.average(madhuri_leverage_row_vals), np.std(madhuri_leverage_row_vals),
            np.average(madhuri_leverage_col_vals), np.std(madhuri_leverage_col_vals),
            np.average(madhuri_leverage_cur_vals), np.std(madhuri_leverage_cur_vals))


def plot_across_singular_vec_dual(ylabel: str, title: str, orig_subspace_data: np.array, orig_leverage_a_data: np.array, orig_leverage_s_data: np.array, pert_subspace_data: np.array, pert_leverage_a_data: np.array, pert_leverage_s_data: np.array, madhuri_subspace_data: np.array, madhuri_leverage_a_data: np.array, madhuri_leverage_s_data: np.array, max_singular_vector_num: int, save_path: str):
    '''Plot CUR and REL_CUR for original, perturbed, and madhuri datasets on a single graph.'''
    plt.errorbar(x=range(0, max_singular_vector_num), y=orig_leverage_a_data, yerr=orig_leverage_s_data, color="blue", linestyle="--", marker="o", capsize=5, label="Original CUR")
    plt.errorbar(x=range(0, max_singular_vector_num), y=pert_leverage_a_data, yerr=pert_leverage_s_data, color="red", linestyle="--", marker="^", capsize=5, label="Perturbed CUR")
    plt.errorbar(x=range(0, max_singular_vector_num), y=madhuri_leverage_a_data, yerr=madhuri_leverage_s_data, color="magenta", linestyle="--", marker="x", capsize=5, label="Perturbed CUR(Madhuri)")

    # Plotting Relative_error_CUR for original, perturbed, and Madhuri datasets
    plt.plot(orig_subspace_data, color="orange", label="Original RELCUR", linestyle='--', marker='>')
    plt.plot(pert_subspace_data, color="green", label="Perturbed RELCUR", linestyle='--', marker='s')
    plt.plot(madhuri_subspace_data, color="cyan", label="Perturbed RELCUR(Madhuri)", linestyle='--', marker='*')

    plt.xticks(range(0, max_singular_vector_num), range(1, max_singular_vector_num+1))
    plt.xlabel("Number of Features Selected")
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    
def plot_curs_dist_dual(original_df: pd.DataFrame, perturbed_df: pd.DataFrame, madhuri_df: pd.DataFrame, max_singular_vector_num: int, constant_search_size: int, leverage_sample_size: int, save_path: str, data_name="Dataset"):
    '''Plot CUR and REL_CUR for both original and perturbed datasets on a single graph.'''
    orig_subspace_col_dist = []
    pert_subspace_col_dist = []
    madhuri_subspace_col_dist = []
    orig_leverage_col_dist = []
    pert_leverage_col_dist = []
    madhuri_leverage_col_dist = []
    orig_subspace_row_dist = []
    pert_subspace_row_dist = []
    madhuri_subspace_row_dist = []
    orig_leverage_row_dist = []
    pert_leverage_row_dist = []
    madhuri_leverage_row_dist = []
    orig_subspace_cur_dist = []
    pert_subspace_cur_dist = []
    madhuri_subspace_cur_dist = []
    orig_leverage_cur_dist = []
    pert_leverage_cur_dist = []
    madhuri_leverage_cur_dist = []
    orig_leverage_col_dist_sd = []
    pert_leverage_col_dist_sd = []
    madhuri_leverage_col_dist_sd = []
    orig_leverage_row_dist_sd = []
    pert_leverage_row_dist_sd = []
    madhuri_leverage_row_dist_sd = []
    orig_leverage_cur_dist_sd = []
    pert_leverage_cur_dist_sd = []
    madhuri_leverage_cur_dist_sd = []

    for i in range(1, max_singular_vector_num+1):
        print(i)
        _, _, _, orig_dists = relerror_cur(original_df, original_df, i, constant_search_size)
        orig_subspace_col_dist.append(orig_dists["col_dist"])
        orig_subspace_row_dist.append(orig_dists["row_dist"])
        orig_subspace_cur_dist.append(orig_dists["cur_dist"])

        _, _, _, pert_dists = relerror_cur(original_df, perturbed_df, i, constant_search_size)
        pert_subspace_col_dist.append(pert_dists["col_dist"])
        pert_subspace_row_dist.append(pert_dists["row_dist"])
        pert_subspace_cur_dist.append(pert_dists["cur_dist"])

        _, _, _, madhuri_dists = relerror_cur(original_df, madhuri_df, i, constant_search_size)
        madhuri_subspace_col_dist.append(madhuri_dists["col_dist"])
        madhuri_subspace_row_dist.append(madhuri_dists["row_dist"])
        madhuri_subspace_cur_dist.append(madhuri_dists["cur_dist"])

        (orig_leverage_row_a, orig_leverage_row_s, orig_leverage_col_a, orig_leverage_col_s, orig_leverage_cur_a, orig_leverage_cur_s,
         pert_leverage_row_a, pert_leverage_row_s, pert_leverage_col_a, pert_leverage_col_s, pert_leverage_cur_a, pert_leverage_cur_s,
         madhuri_leverage_row_a, madhuri_leverage_row_s, madhuri_leverage_col_a, madhuri_leverage_col_s, madhuri_leverage_cur_a, madhuri_leverage_cur_s) = sample_from_leverage_cur_dual(original_df, original_df, perturbed_df, madhuri_df, i, constant_search_size, leverage_sample_size)

        orig_leverage_row_dist.append(orig_leverage_row_a)
        orig_leverage_col_dist.append(orig_leverage_col_a)
        orig_leverage_cur_dist.append(orig_leverage_cur_a)
        orig_leverage_cur_dist_sd.append(orig_leverage_cur_s)
        orig_leverage_col_dist_sd.append(orig_leverage_col_s)
        orig_leverage_row_dist_sd.append(orig_leverage_row_s)

        pert_leverage_row_dist.append(pert_leverage_row_a)
        pert_leverage_col_dist.append(pert_leverage_col_a)
        pert_leverage_cur_dist.append(pert_leverage_cur_a)
        pert_leverage_cur_dist_sd.append(pert_leverage_cur_s)
        pert_leverage_col_dist_sd.append(pert_leverage_col_s)
        pert_leverage_row_dist_sd.append(pert_leverage_row_s)

        madhuri_leverage_row_dist.append(madhuri_leverage_row_a)
        madhuri_leverage_col_dist.append(madhuri_leverage_col_a)
        madhuri_leverage_cur_dist.append(madhuri_leverage_cur_a)
        madhuri_leverage_cur_dist_sd.append(madhuri_leverage_cur_s)
        madhuri_leverage_col_dist_sd.append(madhuri_leverage_col_s)
        madhuri_leverage_row_dist_sd.append(madhuri_leverage_row_s)

    plot_across_singular_vec_dual("RMSE",
                                  f"Fixed Search Size Comparison: RMSE of CUR and REL_CUR Algorithms on {data_name} vs Number of Features Selected",
                                  orig_subspace_cur_dist, orig_leverage_cur_dist, orig_leverage_cur_dist_sd,
                                  pert_subspace_cur_dist, pert_leverage_cur_dist, pert_leverage_cur_dist_sd,
                                  madhuri_subspace_cur_dist, madhuri_leverage_cur_dist, madhuri_leverage_cur_dist_sd,
                                  max_singular_vector_num, save_path + f"{data_name}_RMSECUR.png")


import pandas as pd

def plot_curs_dist_dual_epsilon(original_df: pd.DataFrame, perturbed_df: pd.DataFrame, madhuri_df: pd.DataFrame, max_singular_vector_num: int, constant_search_size: int, leverage_sample_size: int, save_path: str, data_name="Dataset"):
    '''Plot CUR and REL_CUR for original, perturbed, and madhuri datasets on a single graph.'''
    orig_subspace_col_dist = []
    pert_subspace_col_dist = []
    madhuri_subspace_col_dist = []
    orig_leverage_col_dist = []
    pert_leverage_col_dist = []
    madhuri_leverage_col_dist = []
    orig_subspace_row_dist = []
    pert_subspace_row_dist = []
    madhuri_subspace_row_dist = []
    orig_leverage_row_dist = []
    pert_leverage_row_dist = []
    madhuri_leverage_row_dist = []
    orig_subspace_cur_dist = []
    pert_subspace_cur_dist = []
    madhuri_subspace_cur_dist = []
    orig_leverage_cur_dist = []
    pert_leverage_cur_dist = []
    madhuri_leverage_cur_dist = []
    orig_leverage_col_dist_sd = []
    pert_leverage_col_dist_sd = []
    madhuri_leverage_col_dist_sd = []
    orig_leverage_row_dist_sd = []
    pert_leverage_row_dist_sd = []
    madhuri_leverage_row_dist_sd = []
    orig_leverage_cur_dist_sd = []
    pert_leverage_cur_dist_sd = []
    madhuri_leverage_cur_dist_sd = []

    i = max_singular_vector_num  # used for epsilon comparison
    
    _, _, _, orig_dists = relerror_cur(original_df, original_df, i, constant_search_size)
    orig_subspace_col_dist.append(orig_dists["col_dist"])
    orig_subspace_row_dist.append(orig_dists["row_dist"])
    orig_subspace_cur_dist.append(orig_dists["cur_dist"])

    _, _, _, pert_dists = relerror_cur(original_df, perturbed_df, i, constant_search_size)
    pert_subspace_col_dist.append(pert_dists["col_dist"])
    pert_subspace_row_dist.append(pert_dists["row_dist"])
    pert_subspace_cur_dist.append(pert_dists["cur_dist"])

    _, _, _, madhuri_dists = relerror_cur(original_df, madhuri_df, i, constant_search_size)
    madhuri_subspace_col_dist.append(madhuri_dists["col_dist"])
    madhuri_subspace_row_dist.append(madhuri_dists["row_dist"])
    madhuri_subspace_cur_dist.append(madhuri_dists["cur_dist"])

    (orig_leverage_row_a, orig_leverage_row_s, orig_leverage_col_a, orig_leverage_col_s, orig_leverage_cur_a, orig_leverage_cur_s,
     pert_leverage_row_a, pert_leverage_row_s, pert_leverage_col_a, pert_leverage_col_s, pert_leverage_cur_a, pert_leverage_cur_s,
     madhuri_leverage_row_a, madhuri_leverage_row_s, madhuri_leverage_col_a, madhuri_leverage_col_s, madhuri_leverage_cur_a, madhuri_leverage_cur_s) = sample_from_leverage_cur_dual(original_df, original_df, perturbed_df, madhuri_df, i, constant_search_size, leverage_sample_size)

    orig_leverage_row_dist.append(orig_leverage_row_a)
    orig_leverage_col_dist.append(orig_leverage_col_a)
    orig_leverage_cur_dist.append(orig_leverage_cur_a)
    orig_leverage_cur_dist_sd.append(orig_leverage_cur_s)
    orig_leverage_col_dist_sd.append(orig_leverage_col_s)
    orig_leverage_row_dist_sd.append(orig_leverage_row_s)

    pert_leverage_row_dist.append(pert_leverage_row_a)
    pert_leverage_col_dist.append(pert_leverage_col_a)
    pert_leverage_cur_dist.append(pert_leverage_cur_a)
    pert_leverage_cur_dist_sd.append(pert_leverage_cur_s)
    pert_leverage_col_dist_sd.append(pert_leverage_col_s)
    pert_leverage_row_dist_sd.append(pert_leverage_row_s)

    madhuri_leverage_row_dist.append(madhuri_leverage_row_a)
    madhuri_leverage_col_dist.append(madhuri_leverage_col_a)
    madhuri_leverage_cur_dist.append(madhuri_leverage_cur_a)
    madhuri_leverage_cur_dist_sd.append(madhuri_leverage_cur_s)
    madhuri_leverage_col_dist_sd.append(madhuri_leverage_col_s)
    madhuri_leverage_row_dist_sd.append(madhuri_leverage_row_s)

    return orig_subspace_cur_dist, orig_leverage_cur_dist, pert_subspace_cur_dist, pert_leverage_cur_dist, madhuri_subspace_cur_dist, madhuri_leverage_cur_dist
