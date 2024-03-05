In this summer project, we built a novel deterministic CUR decomposition algorithm (Subspace)and compared it against a slightly modified version of an existing non-deterministic method for performing the docomposition.

To test our decomposition, Subspace, on a dataset run:
```
dataframe = some df that is m×n..
top_k_singular_vectors = some number less than min(m,n)..
hyperparameter = some number less than n..
C,U,R,distances = subspace_cur(dataframe, top_k_singular_vectors_to_use, hyperparameter)
```
To test a slighly modified version of the exactly algorithm (here called Leverage) from Drineas Petros, Michael W. Mahoney, and Shan Muthukrishnan's paper ”Relative-error CUR matrix decompositions” that we compare our algorithm against, run:
```
dataframe = some df that is m×n..
top_k_singular_vectors = some number less than min(m,n)..
hyperparameter = some number
C,U,R,distances = leverage_cur(dataframe, top_k_singular_vectors_to_use, hyperparameter)
```
To generate graphs of the performance of Subspace and Leverage over a fixed hyperparemter as the number of singular vectors increases:
```
dataframe = some df that is m×n..
max_top_k_singular_vectors = some number less than min(m,n) that give the maximum number of singular vectors the graph should iterate to.
hyperparameter = some number
sample_leverage_size = some number of samples to run for the leverage algorithm at each point
plot_curs_dist(dataframe, max_top_k_singular_vectors, hyperparameter, sample_leverage_size , "./Figs/dataset_name/", data_name="dataset_name")
```
To get a run time graph of the performance of Subspace and Leverage:
```
plot_curs_time(dataframe, max_top_k_singular_vectors, hyperparameter, sample_leverage_size , "./Figs/dataset_name/", data_name="dataset_name")
```