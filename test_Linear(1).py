import matplotlib.pyplot as plt
import projet_etu as l
import numpy as np
from sklearn import datasets, linear_model
from mltools import plot_data, plot_frontiere, make_grid, gen_arti, load_usps, get_usps, show_usps
from sklearn.metrics import mean_squared_error, r2_score
from Loss import MSELoss


trainx, trainy = datasets.make_regression(n_samples=200, n_features=1, noise=10)
testx, testy = datasets.make_regression(n_samples=200, n_features=1, noise=10)

trainy = np.reshape(trainy, (-1,1))
testy = np.reshape(testy, (-1,1))

linear1 = l.Linear((trainx.shape[1]), (trainy.shape[1]))

mse = MSELoss()

max_iter1 = 1000
for iter in range(max_iter1):
    #Forward
    yhat = linear1.forward(trainx)
    #Loss
    loss = mse.forward(trainy,yhat)
    #Backwards
    bmse = mse.backward(trainy,yhat)
    bl1 = linear1.backward_delta(trainx,bmse)
    
    #Update paramaters and grad 0 
    linear1.backward_update_gradient(trainx, bmse)
    linear1.update_parameters(1e-3)

    linear1.zero_grad()
    
def predict(X):
    yhat = linear1.forward(X)
    return yhat

yhat = predict(testx)
print("accuracy = ", mse.forward(testy, yhat).sum())

plt.figure()
plt.plot(testx, testy, 'g.')
plt.plot(testx, yhat, 'b-', )
plt.show()


# Linear reg from sklearn
lr = linear_model.LinearRegression()
lr.fit(trainx, trainy)

plt.figure()
plt.plot(testx, testy, 'g.')
plt.plot(testx, lr.predict(testx), 'b-', )
plt.show()


