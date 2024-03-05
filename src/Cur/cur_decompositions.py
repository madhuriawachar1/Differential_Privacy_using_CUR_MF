import pandas as pd
import numpy as np
from .subspace import Subspace
from .leverage import Leverage

def subspace_cur(df: pd.DataFrame, sing_vect_num: int, search_size=1)-> tuple[np.array, np.array, np.array, dict]:
    '''Makes a CUR decomosition using the subspace algorithm. Returns the 
    C,U,R matricies and the frobenius distance between the data and CUR, the 
    frobenius norm between the data and the projection onto the C and R matricies.
    df: dataset,
    sing_vect_num: number of top singular vectors to use,
    search_size: search hyperparameter used by subspace'''

    cols, col_distance = Subspace(df).filter(sing_vect_num, search_size)
    rows, row_distance = Subspace(df.T).filter(sing_vect_num,search_size)
    col_matrix = df.loc[:,cols].to_numpy()
    row_matrix = df.loc[rows,:].to_numpy()
    u_matrix = np.linalg.pinv(col_matrix).dot(df).dot(np.linalg.pinv(row_matrix))
    return (col_matrix, u_matrix, row_matrix,
    {"cur_dist": np.linalg.norm(df.to_numpy() - col_matrix.dot(u_matrix).dot(row_matrix), "fro"),"col_dist":col_distance, "row_dist":row_distance})

def leverage_cur(df: pd.DataFrame, sing_vect_num: int, search_size=1) -> tuple[np.array, np.array, np.array, dict]:
    '''Makes a CUR decomosition using the subspace algorithm. Returns the 
    C,U,R matricies and the frobenius distance between the data and CUR, the 
    frobenius norm between the data and the projection onto the C and R matricies.
    df: dataset,
    sing_vect_num: number of top singular vectors to use,
    search_size: search hyperparameter used by leverage'''

    cols, col_distance = Leverage(df).filter(sing_vect_num, search_size)
    rows, row_distance = Leverage(df.T).filter(sing_vect_num,search_size)
    col_matrix = df.loc[:,cols].to_numpy()
    row_matrix = df.loc[rows,:].to_numpy()
    u_matrix = np.linalg.pinv(col_matrix).dot(df).dot(np.linalg.pinv(row_matrix))
    return (col_matrix, u_matrix, row_matrix,
    {"cur_dist":np.linalg.norm(df.to_numpy() - col_matrix.dot(u_matrix).dot(row_matrix)),"col_dist":col_distance, "row_dist":row_distance})


