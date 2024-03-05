from main import plot_curs_dist, plot_curs_time, plot_cur_runtime, leverage_cur, subspace_cur
import pandas as pd
import numpy as np
from sklearn import datasets
def get_gene_data()->pd.DataFrame:
    '''Gets the genetics dataset and removes all the categorical variables
    before returning it as a pandas dataframe.
    '''
    gene_data = pd.read_csv("./genes.csv")
    def is_number(x):
        try:
            float(x)
            return True
        except:
            return False
    is_numeric = np.vectorize(is_number, otypes=[bool])
    return gene_data.loc[:,is_numeric(gene_data.columns)]
    
def O_L2(dataset:pd.DataFrame):
    '''Calculates the largest outlier distance of the 
    L2 norm of the column from the average L2 norm of the columns.'''
    centered = dataset - dataset.mean(axis=0)
    norm = np.linalg.norm(centered, axis=0)
    return max(norm) - np.average(norm)

if __name__ == '__main__':
    print("loading boston")
    boston = pd.DataFrame(datasets.load_boston(return_X_y=True)[0])
    boston = boston - boston.mean(axis=0)
    plot_curs_dist(boston, 13, 10, 15, "./Figs/boston/", data_name="Boston")
    print("finished dist plots for boston")
    plot_curs_time(boston, 13, 10, 15, "./Figs/boston/", data_name="Boston")
    print("finished time plot for boston")
    plot_cur_runtime(boston, 13, 10, 15, 15, "./Figs/boston/", data_name="Boston")
    print("finished runtime plot for boston")
    print(f"Boston O_L2: {O_L2(boston)}")

    print("loading wine")
    wine = pd.DataFrame(datasets.load_wine(return_X_y=True)[0])
    wine = wine - wine.mean(axis=0)
    plot_curs_dist(wine, 13, 10, 15, "./Figs/wine/", data_name="Wine")
    print("finished dist plots for wine")
    plot_curs_time(wine, 13, 10, 35, "./Figs/wine/", data_name="Wine")
    print("finished time plot for wine")
    plot_cur_runtime(wine, 13, 10, 15, 15, "./Figs/wine/", data_name="Wine")
    print("finished runtime plot for wine")
    print(f"Wine O_L2: {O_L2(wine)}")

    print("loading genetics")
    genetics = get_gene_data()
    genetics = genetics - genetics.mean(axis=0)
    plot_curs_dist(genetics, 13, 10, 15, "./Figs/genetics/", data_name="Genes")
    print("finished dist plots for genetics")
    plot_curs_time(genetics, 13, 10, 15, "./Figs/genetics/", data_name="Genes")
    print("finished time plot for genetics")
    plot_cur_runtime(genetics, 13, 10, 15, 15, "./Figs/genetics/", data_name="Genes")
    print("finished runtime plot for genetics")
    print(f"Gene O_L2: {O_L2(boston)}")