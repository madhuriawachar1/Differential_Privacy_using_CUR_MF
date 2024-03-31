import pandas as pd
import numpy as np
from .RelErrorCur import RelErrorCur
from .CUR import CUR

def relerror_cur(df1: pd.DataFrame,df: pd.DataFrame, sing_vect_num: int, search_size=1) -> tuple[np.array, np.array, np.array, dict]:
    '''Makes a CUR decomposition using the subspace algorithm. Returns the 
    C,U,R matrices and the RMSE between the data and CUR, the RMSE between the data and the projection onto the C and R matrices.
    df: dataset,
    sing_vect_num: number of top singular vectors to use,
    search_size: search hyperparameter used by subspace'''

    cols, col_distance = RelErrorCur(df).filter(sing_vect_num, search_size)
    rows, row_distance = RelErrorCur(df.T).filter(sing_vect_num, search_size)
    col_matrix = df.loc[:, cols].to_numpy()
   
    row_matrix = df.loc[rows, :].to_numpy()
    u_matrix = np.linalg.pinv(col_matrix).dot(df).dot(np.linalg.pinv(row_matrix))

    # Calculate RMSE
    cur_approximation = col_matrix.dot(u_matrix).dot(row_matrix)
    cur_rmse = np.sqrt(np.mean((df1.to_numpy() - cur_approximation) ** 2))

    return (col_matrix, u_matrix, row_matrix, {"cur_dist": cur_rmse, "col_dist": col_distance, "row_dist": row_distance})

def normal_cur(df1: pd.DataFrame,df: pd.DataFrame, sing_vect_num: int, search_size=1) -> tuple[np.array, np.array, np.array, dict]:
    
    cols, col_distance = CUR(df).filter(sing_vect_num, search_size)
    col_matrix = df.loc[:, cols].to_numpy()

    # Leverage algorithm for selecting rows
    rows, row_distance = CUR(df.T).filter(sing_vect_num, search_size)
    row_matrix = df.loc[rows, :].to_numpy()

    # Calculate U matrix using pseudoinverse
    u_matrix = np.linalg.pinv(col_matrix).dot(df).dot(np.linalg.pinv(row_matrix))

    # Calculate RMSE
    cur_approximation = col_matrix.dot(u_matrix).dot(row_matrix)
    cur_rmse = np.sqrt(np.mean((df1.to_numpy() - cur_approximation) ** 2))

    return col_matrix, u_matrix, row_matrix, {"cur_dist": cur_rmse, "col_dist": col_distance, "row_dist": row_distance}
