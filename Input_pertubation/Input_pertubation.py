import numpy as np
import pandas as pd
import math
from sklearn.metrics import mean_squared_error
from math import sqrt

import matplotlib.pyplot as plt


def evaluate_item_averages(R, beta_i, epsilon1, epsilon2, r_min, r_max,delta_r):
    n_users, m_items = R.shape  # Assuming R is a matrix of user-item ratings
    print('n_users',n_users,'m_items',m_items )
    
    
    
    # Step 1: Calculate Global Average with Laplace noise
    global_average = np.mean(R)
    global_average += np.random.laplace(loc=0, scale=delta_r/epsilon1)/n_users*m_items

    # Step 2: Initialize Item Averages
    item_averages = np.zeros(m_items)

    # Step 3: Calculate Item Averages with Laplace noise and stabilization
    for i in range(m_items):
        item_ratings = R[:, i]
        item_average = np.mean(item_ratings)
        item_average += beta_i * global_average + np.random.laplace(loc=0, scale=delta_r/epsilon2)
        item_average = np.clip(item_average, r_min, r_max)  # Clamp to the range [r_min, r_max]

        item_averages[i] = item_average

    return global_average, item_averages



def evaluate_user_averages(R, item_averages, B, epsilon1, epsilon2, u_min, u_max,delta_r):
    n_users, m_items = R.shape

    # Step 1: Calculate R'
    R_discounted = R - np.outer(np.ones(n_users), item_averages)

    # Step 2: Calculate Global Average with Laplace noise
    global_average = np.mean(R_discounted)
    global_average += np.random.laplace(loc=0, scale=delta_r/epsilon1)/n_users*m_items
    # Step 3: Initialize User Averages
    user_averages = np.zeros(n_users)

    # Step 4: Calculate User Averages with Laplace noise and stabilization
    for v in range(n_users):
        user_ratings = R_discounted[v, :]
        user_average = np.mean(user_ratings)
        user_average += B * global_average + np.random.laplace(loc=0, scale=delta_r/epsilon2)
        user_average = np.clip(user_average, u_min, u_max)

        user_averages[v] = user_average

    return user_averages

def differentially_private_input_perturbation(R, B, epsilon,delta_r):
    # Step 1: Add Laplace noise to ratings in R
    R_perturbed = R + np.random.laplace(scale=(1/epsilon), size=R.shape)

    # Step 2: Clamp the ratings in R' to the range [-B, B]
    R_perturbed = np.clip(R_perturbed, -B, B)
    return R_perturbed

