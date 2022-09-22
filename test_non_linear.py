import projet_etu as l
import numpy as np
from mltools import plot_data, plot_frontiere, make_grid, gen_arti, load_usps, get_usps, show_usps
import matplotlib.pyplot as plt
from Loss import MSELoss

# (0 ,1)

trainx, trainy = gen_arti(nbex = 500, data_type=1, epsilon = 0.1)
testx, testy = gen_arti(nbex = 500, data_type=1, epsilon = 0.1)

trainy = np.where(trainy < 0, 0 ,1)
testy = np.where(testy < 0, 0 ,1)


linear1 = l.Linear((trainx.shape[1]), 20)
tan = l.TanH()
linear2 = l.Linear(20, (trainy.shape[1]))
sig = l.Sigmoide()
mse = MSELoss()

max_iter1 = 100
for iter in range(max_iter1):

    #Forward
    z1 = linear1.forward(trainx)
    z2 = tan.forward(z1)
    z3 = linear2.forward(z2)
    yhat = sig.forward(z3)

    #Loss
    loss = mse.forward(trainy,yhat)

    #print("Linear Iteration ",iter," : => LOSS : ", loss.sum())

    #Backwards
    bmse = mse.backward(trainy,yhat)
    
    bsig = sig.backward_delta(z3, bmse)
    bl2 = linear2.backward_delta(z2,bsig)
    bt1 = tan.backward_delta(z1, bl2)
    bl1 = linear1.backward_delta(trainx,bt1)

    #Update paramaters and grad 0 

    linear2.backward_update_gradient(z2, bsig)
    linear1.backward_update_gradient(trainx, bt1)

    linear2.update_parameters(1e-3)
    linear1.update_parameters(1e-3)

    linear2.zero_grad()
    linear1.zero_grad()

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