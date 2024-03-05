import pandas as pd
import numpy as np
class FeatureSelector ():
    def __init__(self,labeled_data:pd.DataFrame,mean_center=False):
        '''Gives a general parent class to be used by both 
        the Leverage and Subspace algorithms. It centers the data, 
        drops dependant variables, and sets up the distance and 
        selected features object variables. '''
        if "Classification" in labeled_data.columns:
            self.data = labeled_data.drop("Classification", axis=1)
        else:
            self.data = labeled_data  
        if mean_center:
            self.data = self.data - self.data.mean(axis=0)
        self.space_dist = None
        self.selected_features = None
def make_projection(C):
    ''' builds a projection matrix'''
    return C.dot(np.linalg.pinv(C))

def subspace_dist(X,C):
    ''' calculates the frobenius norm of the projection of the dataset X
    onto C from the original dataset'''
    return np.linalg.norm(X - make_projection(C).dot(X),ord="fro")
