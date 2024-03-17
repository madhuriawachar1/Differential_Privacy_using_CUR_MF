import numpy as np
import pandas as pd
import time
from typing import Callable

from ..Cur.rel_cur_decomposition import *
import matplotlib.pyplot as plt
from ..Cur.cur_decomposition import *

#def plot_across_singular_vec(ylabel:str, title:str, subspace_data:np.array, leverage_a_data:np.array,leverage_s_data:np.array, max_singular_vector_num:int, save_path:str):
    
def plot_across_singular_vec(ylabel:str, title:str, subspace_data:np.array, leverage_a_data:np.array, leverage_s_data:np.array, max_singular_vector_num:int, save_path:str):
    ''' helper func for plot_curs_dist, plots the standard layout for comparing the distances
    between subspace and leverage.
    '''
    #plt.errorbar(x=range(0,max_singular_vector_num),y=leverage_a_data,yerr = leverage_s_data,color="red",linestyle="--",marker="o",capsize=5,label="CUR")
    plt.plot(subspace_data,color="blue",label="rel_cur",linestyle='--', marker='o')
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
    
    subspace_cur_dist = []
    leverage_cur_dist = []
    
    #leverage_cur_dist_sd = []
    for i in range(1,max_singular_vector_num+1):
        print(i)
        _, _, _, dists = subspace_cur(df,i,constant_search_size)
        
        subspace_cur_dist.append(dists["cur_dist"])
        a=CUR(df.to_numpy(), 100,i)
        _, _, _, dists = a.cur()
        leverage_cur_dist.append(dists["cur_dist"])
    
    plot_across_singular_vec("RMSE",
        f"Fixed Search Size Comparison: \n CUR Frobenius Norm of rel_cur and cur Algorithms \non {data_name} vs Number of Features Selected",
        subspace_cur_dist,
        leverage_cur_dist,
        np.std(leverage_cur_dist),
        max_singular_vector_num,
        save_path + f"{data_name}_CUR.png")
    