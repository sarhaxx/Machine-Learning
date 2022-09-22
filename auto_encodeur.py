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


#===== MNIST =========

(train_X, y), (test_X, yh) = mnist.load_data()

train_X = train_X.reshape(train_X.shape[0], train_X.shape[1]**2) / 255.0
test_X = test_X.reshape(test_X.shape[0], test_X.shape[1]**2) / 255.0


test_X =test_X[:10000]

# ======================

linear1 = nn.Linear(784,512)
lin1    = nn.Linear(512,128)
linear2 = nn.Linear(128,10)
linear3 = nn.Linear(10,128)
lin3    = nn.Linear(128,512)
linear4 = nn.Linear(512,784)

######

linear1 = nn.Linear(784,100)
linear2 = nn.Linear(100,10)
linear3 = nn.Linear(10,100)
linear4 = nn.Linear(100,784)


linear9 = nn.Linear(784,2500)
linear10 = nn.Linear(2500, 2000)
linear11 = nn.Linear(2000, 1500)
linear12 = nn.Linear(1500,1000)
linear13 = nn.Linear(1000,500)
linear14 = nn.Linear(500, 10)

linear15 = nn.Linear(10,500)
linear16 = nn.Linear(500, 1000)
linear17 = nn.Linear(1000, 1500)
linear18 = nn.Linear(1500,2000)
linear19 = nn.Linear(2000,2500)
linear20 = nn.Linear(2500, 784)


# 256 -> 100 > 10 > 100 > 

sig = nn.Sigmoide()
tan = nn.TanH()
softmax  = nn.softmax()

L = [linear9,tan, linear10, tan, linear11, linear12, tan, linear13, tan, linear14, linear15, tan, linear16, tan,linear17, linear18, tan, linear19, tan, linear20, sig]


"""
class autoencodeur(Module):
    def __init__(self, latent_dim, seq_encoder, seq_decoder):
        self.latent_dim = latent_dim
        self.seq_encoder = seq_encoder
        self.seq_decoder = seq_decoder

    def encoder(self, input):
        encoded = self.seq_encoder.layers[0].forward(input)
        for i in range(1,len(self.seq_encoder.layers)):
            encoded = self.seq_encoder.layers[i].forward(encoded)
        return self.seq_encoder.forward(input)

    def decoder(self, input):
        decoded = self.seq_decoder.layers[0].forward(input)
        for i in range(1,len(self.seq_decoder.layers)):
            decoded = self.seq_decoder.layers[i].forward(decoded)
        return self.seq_decoder.forward(input)

    def forward(self, input):
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)
        
        return decoded

    def backward(self, delta):
        d = self.seq_encoder.backward( delta)
        d1= self.seq_decoder.backward( d) 

    def zero_grad(self):
        self.seq_encoder.zero_grad()
        self.seq_decoder.zero_grad()
    
    def upadate_parameters(self,delta):
        self.seq_decoder.update_parameters()
        self.seq_encoder.update_parameters()
"""

loss = BCELoss()
hist_loss = []



print("TRAIN.......")


L1 = [linear1, tan, linear2, tan]
L2 = [linear3, tan, linear4, sig]
L = L1 + L2
seq = nn.Sequentiel(L)

hist_loss = nn.SGD(seq, train_X , train_X, 100, 400, loss, eps = 1e-4)
X = seq.forward(test_X).reshape(-1,28,28)

print("TEST..........")

X_visu = []
for i in range(len(test_X)):
    plt.imshow(test_X.reshape(-1,28,28)[i])
    plt.show()
    plt.imshow(X[i])
    plt.show()



plt.figure()
plt.plot(list(range(100)) , hist_loss, 'r-')
plt.show()

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
