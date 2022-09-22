import projet_etu as l
import numpy as np
from mltools import plot_data, plot_frontiere, make_grid, gen_arti, load_usps, get_usps, show_usps
import matplotlib.pyplot as plt
from Loss import MSELoss

trainx, trainy = gen_arti(nbex = 500, data_type=1, epsilon = 0.1)
testx, testy = gen_arti(nbex = 500, data_type=1, epsilon = 0.1)

trainy = np.where(trainy < 0, 0 ,1)
testy = np.where(testy < 0, 0 ,1)


linear1 = l.Linear((trainx.shape[1]), 20)
tan = l.TanH()
linear2 = l.Linear(20, (trainy.shape[1]))
sig = l.Sigmoide()
mse = MSELoss()

L = [linear1,tan,linear2,sig] 
seq = l.Sequentiel(L)

max_iter2 = 0
for iter in range(max_iter2):
    #Calcul forward
    op = l.Optim(seq, mse, 1e-3)
    op.step(trainx, trainy)


l.SGD(seq,trainx,trainy, 5, 10000, mse, 1e-3)

def predict(X):
    z1 = linear1.forward(X)
    z2 = tan.forward(z1)
    z3 = linear2.forward(z2)
    yhat = sig.forward(z3)
    return np.where(yhat >= 0.5, 1, 0)

print("accuracy = ", np.where(testy == predict(testx) , 1 , 0).sum()/len(testx))

plot_frontiere(testx, predict ,step=200)
plot_data(testx,testy)
plt.show()







