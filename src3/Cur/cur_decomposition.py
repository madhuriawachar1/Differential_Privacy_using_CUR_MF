import pandas as pd
import numpy as np
import math
import pickle


#ref:https://github.com/iamkroot/recommender-systems/tree/master


def get_user_movie_rating_matrix(file_path):
    
    df = pd.read_csv(file_path, sep="\t", names=['user_id', 'movie_id', 'rating', 'timestamp'])
    num_users = df['user_id'].max()
    num_movies = df['movie_id'].max()

    user_movie_rating_matrix = np.zeros((num_users, num_movies))

    for i in range(len(df)):
        user_id = int(df['user_id'][i]) - 1
        movie_id = int(df['movie_id'][i]) - 1
        rating = float(df['rating'][i])

        user_movie_rating_matrix[user_id][movie_id] = rating

    return user_movie_rating_matrix



class CUR():
    """
        Predicts the ratings of first quater of user movie matrix using CUR
    """

    def __init__(self, rating_matrix,epochs,max_singular_vector_num, num_factors=50, learning_rate=0.01, regularization=0.1):
        self.rating_matrix = rating_matrix
        self.num_users, self.num_movies = rating_matrix.shape
        self.num_factors = num_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
        self.max_singular_vector_num=max_singular_vector_num
        # Initialize matrices C, U, and R with random values
        self.C = np.random.rand(self.num_users, num_factors)
        self.U = np.random.rand(num_factors, self.num_movies)
        self.R = np.random.rand(num_factors, self.num_movies)

        self.generated_rating_matrix = self.cur()

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
    

    def cur(self):
        sample_size = self.max_singular_vector_num

        # Sampling columns - C
        matrix_sum = (self.rating_matrix**2).sum()
        col_prob = (self.rating_matrix**2).sum(axis=0)
        col_prob /= matrix_sum

        col_indices = np.random.choice(np.arange(0,self.num_movies), size=sample_size, replace=True, p=col_prob)
        self.C = self.rating_matrix.copy()[:,col_indices]
        self.C = np.divide(self.C,(sample_size*col_prob[col_indices])**0.5)

        # Sampling rows - R
        row_prob = (self.rating_matrix**2).sum(axis=1)
        row_prob /= matrix_sum

        row_indices = np.random.choice(np.arange(0,self.num_users), size=sample_size, replace=True, p=row_prob)
        self.R = self.rating_matrix.copy()[row_indices, :]
        self.R = np.divide(self.R, np.array([(sample_size*row_prob[row_indices])**0.5]).transpose())

        # Finding U

        # W - intersection of sampled C and R
        W = self.rating_matrix.copy()[:, col_indices]
        W = W[row_indices, :]

        X, Z, YT = self.svd(W,self.max_singular_vector_num)

        for i in range(min(Z.shape[0],Z.shape[1])):
            if (Z[i][i] != 0):
                Z[i][i] = 1/Z[i][i]

        Y = YT.transpose()
        XT = X.transpose()

        self.U = Y.dot(Z.dot(XT))
        reconstructed_matrix = self.C.dot(self.U).dot(self.R)
        return (self.C, self.U, self.R,
        {"cur_dist": np.sqrt(np.mean((self.rating_matrix - reconstructed_matrix)**2))})

  