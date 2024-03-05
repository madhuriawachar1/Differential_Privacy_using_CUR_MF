import numpy as np
from .featureSelector import FeatureSelector, subspace_dist
def rearrange_indexes(index_arr:np.array):
    '''Reorders the indicies selected by parallel_alg where
    deleted elements of X shift the remaining indexes. The indexes
    in terms of the original X are returned'''
    for i in range(len(index_arr)-1,-1,-1):
        for j in range(len(index_arr)-1,i,-1):
            if index_arr[j] >= index_arr[i]:
                index_arr[j] += 1

def alg_c_optimize(X:np.array,k:int,c:int):
    ''' Performs the global computations used across
    all c choices. Sets up the c choices in a way that they 
    could be easily parallelized for large datasets. X is the dataset,
    k is the number of singular vectors to use, c is the number of selections 
    to minimize over. Returns the set of all indexes from each selection as well
    as the computed k left singular vectors. '''
    U_k = np.linalg.svd(X)[0][:,:k]
    X_norm = X/np.linalg.norm(X,axis=0)
    epsilons = np.linalg.norm(np.dot(U_k.T,X_norm),axis=0)**2
    if c < len(epsilons):
        initial_choices = np.argpartition(epsilons,-c)[-c:]
    else:
        initial_choices = np.argsort(epsilons)[-c:]
    indexes = np.empty((k,c),dtype=int)
    for i,choice in enumerate(initial_choices):
        indexes[:,i] = parallel_alg(X_norm.copy(), choice, U_k, epsilons.copy(), k)
    return (indexes,U_k)

def parallel_alg(X_norm:np.array, initial_index:int, U_k:np.array, epsilons:np.array,k:int,return_error=False):
    ''' Performs the remaining computations needed to perform subspace. Generates a set of 
    indexes from a starting choice given by the global alg_c_optimize. X_norm is X normalized
    over its columns, initial_index gives the starting index for the algorithm, U_k gives the 
    k left singular vectors, epsilons gives the initial distance each point is from the total 
    U_k subspace, k gives the number of left singular vectors to use.'''
    deltas = np.empty((X_norm.shape[0],k),dtype=float)
    chosen_indexes = np.empty(k,dtype=int)
    errors = np.empty(k,dtype=float)
    for i in range(k):
        if i == 0:
            min_index = initial_index
            errors[i] = epsilons[min_index]
        else:
            min_index = np.argmax(epsilons)
            errors[i] = epsilons[min_index]
        p = np.dot(np.dot(U_k,U_k.T), X_norm[:,min_index])
        for j in range(0,i):
            p = p - np.dot(p,deltas[:,j])
        deltas[:,i] = p/np.linalg.norm(p)
        epsilons = epsilons - np.linalg.norm(np.dot(deltas[:,i],X_norm),axis=0)**2
        chosen_indexes[i] = min_index
        epsilons = np.delete(epsilons,min_index,axis = 0)
    rearrange_indexes(chosen_indexes)
    if return_error:
        return chosen_indexes, errors
    return chosen_indexes

class Subspace(FeatureSelector):
    def filter(self,k:int,c:int):
        '''Performs the algorithm Subspace from the paper. Takes in 
        k, the total number of left singular vectors to use and c,
        the total number of index choices to minimize the Proj Distance over.
        Returns the selected features and the distance away it is.
        
        A reasonable suggestion for k and c would be the total number of columns in your
        dataframe divided by 4'''
        if k == 0:
            raise Exception("singular vector number must be postive int")
        if k > len(self.data.index):
            raise Exception("cannot use more singular vectors than the row size")
        if c > len(self.data.columns):
            raise Exception("cannot search through a size greater than the number of columns")
        X = self.data.to_numpy()
        indexes,_ = alg_c_optimize(X,k,c)
        dist_arr = np.empty(c,dtype=float)
        for i,col in enumerate(indexes.T):
            dist_arr[i] = subspace_dist(X,X[:,col])
        smallest_dist = np.argmin(dist_arr)
        best_index = indexes[:,smallest_dist]
        self.space_dist = dist_arr[smallest_dist]
        self.selected_features = self.data.columns[best_index]
        return self.selected_features,self.space_dist