import projet_etu as l
from mltools import plot_data, plot_frontiere, make_grid, load_usps, get_usps, show_usps
import numpy as np
import matplotlib.pyplot as plt
from Loss import BCELoss, MSELoss
import projet_etu  as nn
from keras.datasets import mnist
import torchvision
from sklearn.utils import shuffle
import time 
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

(train_X, y), (test_X, yh) = mnist.load_data()

X = train_X.reshape(train_X.shape[0], train_X.shape[1]**2) / 255.0


def t_sne(X,y):
    feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
    df = pd.DataFrame(X,columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))
    X, y = None, None

    np.random.seed(42)
    rndperm = np.random.permutation(df.shape[0])

    N = 10000
    df_subset = df.loc[rndperm[:N],:].copy()
    data_subset = df_subset[feat_cols].values

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data_subset)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


    df_subset['tsne-2d-one'] = tsne_results[:,0]
    df_subset['tsne-2d-two'] = tsne_results[:,1]
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", 10),
        data=df_subset,
        legend="full",
        alpha=0.3
    )
    plt.show()


t_sne(X,y)