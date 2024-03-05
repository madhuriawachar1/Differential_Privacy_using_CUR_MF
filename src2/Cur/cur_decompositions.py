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
    def svd(self, matrix, k):
        """
        Performs the SVD decomposition on the input matrix

        Args:
            matrix (np.ndarray) : The user rating matrix
            k (int) : the reduced dimensionality after decomposition

        Returns:
            The three SVD matrices U,Sigma and V_T

        """
        m = matrix.shape[0]
        n = matrix.shape[1]

        if (k > m) or (k > n):
            print("error: k greater than matrix dimensions.\n")
            return

        matrix_t = matrix.T

        A = np.dot(matrix, matrix_t)  # calculate matrix multiplied by its transpose
        values1, v1 = np.linalg.eigh(A)  # get eigenvalues and eigenvectors
        v1_t = v1.T
        # discarding negative eigenvalues and corresponding eigenvectors (they are anyway tending to zero)
        v1_t[values1 < 0] = 0
        v1 = v1_t.T
        values1[values1 < 0] = 0
        # values1 = np.absolute(values1)

        values1 = np.sqrt(values1)  # finding singular values.
        # sort eigenvalues and eigenvectors in decreasing order
        idx = np.argsort(values1)
        idx = idx[::-1]
        values1 = values1[idx]
        v1 = v1[:, idx]

        U = v1

        A = np.dot(matrix_t, matrix)  # calculate matrix transpose multiplied by matrix.
        values2, v2 = np.linalg.eigh(A)  # get eigenvalues and eigenvectors
        # values2 = np.absolute(values2)
        # discarding negative eigenvalues and corresponding eigenvectors(they are anyway tending to zero)
        v2_t = v2.T
        v2_t[values2 < 0] = 0
        v2 = v2_t.T
        values2[values2 < 0] = 0

        values2 = np.sqrt(values2)  # finding singular values.
        # sort eigenvalues and eigenvectors in decreasing order.
        idx = np.argsort(values2)
        idx = idx[::-1]
        values2 = values2[idx]
        v2 = v2[:, idx]

        V = v2
        V_t = V.T  # taking V transpose.

        sigma = np.zeros((m, n))

        if m > n:  # setting the dimensions of sigma matrix.

            sigma[:n, :] = np.diag(values2)

        elif n > m:
            sigma[:, :m] = np.diag(values1)

        else:
            sigma[:, :] = np.diag(values1)

        if m > k:  # slicing the matrices according to the k value.
            U = U[:, :k]
            sigma = sigma[:k, :]

        if n > k:
            V_t = V_t[:k, :]
            sigma = sigma[:, :k]

        check = np.dot(matrix, V_t.T)
        # case = np.divide(check, values2[:k])

        s1 = np.sign(check)
        s2 = np.sign(U)
        c = s1 == s2
        # choosing the correct signs of eigenvectors in the U matrix.
        for i in range(U.shape[1]):
            if c[0, i] is False:
                U[:, i] = U[:, i] * -1

        return U, sigma, V_t
    sample_size = sing_vect_num

    # Sum of squares of all elements
    matrix_sum = (df**2).sum()

    # Sampling columns - C
    col_prob = matrix_sum / matrix_sum.sum()
    col_indices = np.random.choice(df.shape[1], size=sample_size, replace=True, p=col_prob)
    col_matrix = df.iloc[:, col_indices].to_numpy()
    col_matrix /= np.sqrt(sample_size * col_prob[col_indices])

    # Sampling rows - R
    row_prob = matrix_sum / matrix_sum.sum()
    row_indices = np.random.choice(df.shape[0], size=sample_size, replace=True, p=row_prob)
    row_matrix = df.iloc[row_indices, :].to_numpy()
    row_matrix /= np.sqrt(sample_size * row_prob[row_indices])

    # Calculate U matrix
    W = df.iloc[row_indices, col_indices].to_numpy()
    U, _, _ = svd(W, 50)  # assuming svd is defined elsewhere in your code

    # Calculate Frobenius distance
    cur_dist = np.linalg.norm(df.to_numpy() - np.dot(col_matrix, np.dot(U, row_matrix)))
    col_dist = np.linalg.norm(df.to_numpy() - np.dot(col_matrix, np.linalg.pinv(col_matrix).dot(df)))
    row_dist = np.linalg.norm(df.to_numpy() - np.dot(np.linalg.pinv(row_matrix).dot(df), row_matrix))

    return col_matrix, U, row_matrix, {"cur_dist": cur_dist, "col_dist": col_dist, "row_dist": row_dist}