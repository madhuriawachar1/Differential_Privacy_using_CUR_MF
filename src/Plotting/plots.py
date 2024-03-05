import numpy as np
import pandas as pd
import time
from typing import Callable

from ..Cur.cur_decompositions import *
import matplotlib.pyplot as plt

def sample_from_leverage_cur(df:pd.DataFrame, sing_vect_num:int, search_size:int, sample_size=50):
    '''helper func for plot_curs_dist. Samples from the leverage cur decomposition
     a number of times equal to samples_size to generate mean and standard dev.
     distances. Inputs df, sing_vect_num, search_size into leverage_cur for the computation
    '''
    leverage_row_vals = np.empty(sample_size)
    leverage_col_vals = np.empty(sample_size)
    leverage_cur_vals = np.empty(sample_size)
    for i in range(sample_size):
        _,_,_,dists = leverage_cur(df, sing_vect_num, search_size)
        leverage_row_vals[i] = dists["row_dist"]
        leverage_col_vals[i] = dists["col_dist"]
        leverage_cur_vals[i] = dists["cur_dist"]
    return (np.average(leverage_row_vals), np.std(leverage_row_vals),
    np.average(leverage_col_vals), np.std(leverage_col_vals),
    np.average(leverage_cur_vals), np.std(leverage_cur_vals))


def plot_across_singular_vec(ylabel:str, title:str, subspace_data:np.array, leverage_a_data:np.array, leverage_s_data:np.array, max_singular_vector_num:int, save_path:str):
    ''' helper func for plot_curs_dist, plots the standard layout for comparing the distances
    between subspace and leverage.
    '''
    plt.errorbar(x=range(0,max_singular_vector_num),y=leverage_a_data,yerr = leverage_s_data,color="red",linestyle="--",marker="o",capsize=5,label="Leverage")
    plt.plot(subspace_data,color="blue",label="Subspace",linestyle='--', marker='o')
    plt.xticks(range(0,max_singular_vector_num),range(1,max_singular_vector_num+1))
    plt.xlabel("Number of Features Selected")
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)
    plt.savefig(save_path,bbox_inches="tight")
    plt.close()

def plot_curs_dist(df: pd.DataFrame, max_singular_vector_num:int, constant_search_size:int, leverage_sample_size:int, save_path:str, data_name="Dataset"):
    '''plots the frobenius norm between the dataset and the projection onto the columns
    and rows of the C, R matricies for both the leverage and subspace decomopsitions. 
    Also plots the frobenius norm of the dataset and the full cur decomposition.'''
    subspace_col_dist = []
    leverage_col_dist = []
    subspace_row_dist = []
    leverage_row_dist = []
    subspace_cur_dist = []
    leverage_cur_dist = []
    leverage_col_dist_sd = []
    leverage_row_dist_sd = []
    leverage_cur_dist_sd = []
    for i in range(1,max_singular_vector_num+1):
        print(i)
        _, _, _, dists = subspace_cur(df,i,constant_search_size)
        subspace_col_dist.append(dists["col_dist"])
        subspace_row_dist.append(dists["row_dist"])
        subspace_cur_dist.append(dists["cur_dist"])
        (leverage_row_a, leverage_row_s, leverage_col_a, 
        leverage_col_s, leverage_cur_a, leverage_cur_s) = sample_from_leverage_cur(df, i, constant_search_size, leverage_sample_size)
        leverage_row_dist.append(leverage_row_a)
        leverage_col_dist.append(leverage_col_a)
        leverage_cur_dist.append(leverage_cur_a)
        leverage_cur_dist_sd.append(leverage_cur_s)
        leverage_col_dist_sd.append(leverage_col_s)
        leverage_row_dist_sd.append(leverage_row_s)
    plot_across_singular_vec("Frobenius Error of Data and CUR Approximation",
        f"Fixed Search Size Comparison: \n CUR Frobenius Norm of Subspace and Leverage Algorithms \non {data_name} vs Number of Features Selected",
        subspace_cur_dist,
        leverage_cur_dist,
        leverage_cur_dist_sd,
        max_singular_vector_num,
        save_path + f"{data_name}_CUR.png")
    plot_across_singular_vec("Projection Frobenius Error",
        f"Fixed Search Size Comparison: \n Projection Distance of Subspace and Leverage Algorithms \non {data_name}'s Columns vs Number of Features Selected",
        subspace_col_dist,
        leverage_col_dist,
        leverage_col_dist_sd,
        max_singular_vector_num,
        save_path + f"{data_name}_Col.png")
    plot_across_singular_vec("Projection Frobenius Error",
        f"Fixed Search Size Comparison: \n Projection Distance of Subspace and Leverage Algorithms \non {data_name}'s Rows vs Number of Features Selected",
        subspace_row_dist,
        leverage_row_dist,
        leverage_row_dist_sd,
        max_singular_vector_num,
        save_path + f"{data_name}_Row.png")

def time_func(func : Callable, *args):
    '''Measures the time taken for a function to run. Adds the args
    to the func before running.'''
    start = time.time()   
    _ = func(*args)
    return time.time() - start

def avg_sd_time(func : Callable, iterations:int,*args):
    '''Measures the average and standard deviation of times taken for 
    a function to run over a number of iterations. Adds in args
    into function each time before running.'''
    times = np.empty(iterations)
    for i in range(iterations):
        times[i] = time_func(func, *args)
    return np.average(times), np.std(times)

def plot_curs_time(df:pd.DataFrame, max_singular_vect_num:int, constant_search_size:int, iterations:int, save_path:str, data_name="Dataset"):
    '''Plots the time taken for the CUR decomposition to run on both subspace
    and leverage. Outputs into the figs folder.'''
    subspace_a_arr = np.empty(max_singular_vect_num)
    subspace_sd_arr = np.empty(max_singular_vect_num)
    leverage_a_arr = np.empty(max_singular_vect_num)
    leverage_sd_arr = np.empty(max_singular_vect_num)
    for i in range(1,max_singular_vect_num+1):
        print(i)
        subspace_a, subspace_sd = avg_sd_time(subspace_cur,iterations,df,i, constant_search_size)
        leverage_a, leverage_sd = avg_sd_time(leverage_cur,iterations,df,i, constant_search_size)
        subspace_a_arr[i-1] = subspace_a
        subspace_sd_arr[i-1] = subspace_sd
        leverage_a_arr[i-1] = leverage_a
        leverage_sd_arr[i-1] = leverage_sd
    plt.errorbar(x=range(0,max_singular_vect_num),y=subspace_a_arr,yerr= subspace_sd_arr, color="blue",label="Subspace",linestyle="--",marker="o",capsize=5)
    plt.errorbar(x=range(0,max_singular_vect_num),y=leverage_a_arr,yerr = leverage_sd_arr, color="red",label="Leverage",linestyle="--",marker="o",capsize=5)
    plt.xticks(range(0,max_singular_vect_num),range(1,max_singular_vect_num+1))
    plt.xlabel("Number of Features Selected")
    plt.ylabel("Runtime (s)")
    plt.legend()
    plt.title(f"Runtime of Subspace and Leverage Algorithm's CUR \non {data_name} vs Number of Features Selected")
    plt.savefig(save_path + f"{data_name}_Time.png")
    plt.close()

def plot_cur_runtime(df:pd.DataFrame, max_singular_vect_num:int, constant_search_size:int, leverage_sample_size:int, iterations:int, save_path:str, data_name="Dataset"):
    subspace_col_dist = []
    leverage_col_dist = []
    subspace_row_dist = []
    leverage_row_dist = []
    subspace_cur_dist = []
    leverage_cur_dist = []
    leverage_col_dist_sd = []
    leverage_row_dist_sd = []
    leverage_cur_dist_sd = []
    for i in range(1, max_singular_vect_num+1):
        print(i)
        subspace_av_time,_ = avg_sd_time(subspace_cur,iterations, df, i, constant_search_size)
        leverage_av_time = 0
        hyper_search = 0
        while leverage_av_time < subspace_av_time:
            hyper_search += 1
            leverage_av_time, _ = avg_sd_time(leverage_cur,iterations, df, i, hyper_search)
        
        _, _, _, dists = subspace_cur(df,i,constant_search_size)
        subspace_col_dist.append(dists["col_dist"])
        subspace_row_dist.append(dists["row_dist"])
        subspace_cur_dist.append(dists["cur_dist"])
        (leverage_row_a, leverage_row_s, leverage_col_a, 
        leverage_col_s, leverage_cur_a, leverage_cur_s) = sample_from_leverage_cur(df, i, hyper_search, leverage_sample_size)
        leverage_row_dist.append(leverage_row_a)
        leverage_col_dist.append(leverage_col_a)
        leverage_cur_dist.append(leverage_cur_a)
        leverage_cur_dist_sd.append(leverage_cur_s)
        leverage_col_dist_sd.append(leverage_col_s)
        leverage_row_dist_sd.append(leverage_row_s)

    plot_across_singular_vec("Frobenius Error of Data and CUR Approximation",
        f"Runtime Comparison:\n CUR Frobenius Norm of Subspace and Leverage Algorithms \non {data_name} vs Number of Features Selected",
        subspace_cur_dist,
        leverage_cur_dist,
        leverage_cur_dist_sd,
        max_singular_vect_num,
        save_path + f"{data_name}_Runtime_CUR.png")
    plot_across_singular_vec("Projection Frobenius Error",
        f"Runtime Comparison:\n Projection Distance of Subspace and Leverage Algorithms \non {data_name}'s Columns vs Number of Features Selected",
        subspace_col_dist,
        leverage_col_dist,
        leverage_col_dist_sd,
        max_singular_vect_num,
        save_path + f"{data_name}_Runtime_Col.png")
    plot_across_singular_vec("Projection Frobenius Error",
        f"Runtime Comparison: \n Projection Distance of Subspace and Leverage Algorithms \non {data_name}'s Rows vs Number of Features Selected",
        subspace_row_dist,
        leverage_row_dist,
        leverage_row_dist_sd,
        max_singular_vect_num,
        save_path + f"{data_name}_Runtime_Row.png")