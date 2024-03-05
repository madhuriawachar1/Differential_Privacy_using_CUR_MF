import numpy as np
from .featureSelector import FeatureSelector, subspace_dist
class Leverage (FeatureSelector):
    def filter(self,k: int, c: int):
        '''Performs leverage sampling as described in the paper.
        k gives the number of singular vectors to use and c gives the 
        number of indicie sets to generate and minimize the Proj Distance 
        over. Returns the selected features and the distance away it is. 

        IMPORTANT NOTE: unlike in the malhalony et all paper, the columns selected are not
        rescaled. For the purposes of this paper, it was an unnecessary addition since the 
        comparisons using this algorithm all have a projection of X onto C which is independant
        of a resaling of the matrix C.
        '''
        if k == 0:
            raise Exception("singular vector number must be postive int")
        if k > self.data.shape[1]:
            raise Exception("cannot choose more singular vectors than columns")
        data = self.data.to_numpy()
        _,_,V_t = np.linalg.svd(data)
        V = V_t.T
        def leverage(V,k,j):
            return np.dot(V[j,:k].T,V[j,:k])
        col_leverage = np.array([leverage(V,k,j)/k for j in range(data.shape[1])])
        sampled = np.random.choice(np.arange(0,data.shape[1],dtype=int),size=(c,k),replace = True, p=col_leverage)
        dist_arr = [subspace_dist(data,data[:,col]) for col in sampled]
        smallest_dist = np.argmin(dist_arr)
        best_index = sampled[smallest_dist,:]
        self.space_dist = dist_arr[smallest_dist]
        self.selected_features = self.data.columns[best_index]
        return self.selected_features,self.space_dist

    def highest_leverage(self,k,c):
        '''finds the c highest leverage columns with respect to the
        first k singular vectors.'''
        data = self.data.to_numpy()
        _,_,V_t = np.linalg.svd(data)
        V = V_t.T
        def leverage(V,k,j):
            return np.dot(V[j,:k].T,V[j,:k])
        col_leverage = np.array([leverage(V,k,j)/k for j in range(data.shape[1])])
        return self.data.columns[np.argsort(col_leverage)[len(col_leverage)-c:]]
