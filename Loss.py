from ast import Break
from turtle import forward
import numpy as np 


class Loss(object):
    def forward(self, y, yhat):
        pass 

    def backward(self, y, yhat):
        pass        

class MSELoss(Loss):

    def forward(self, y, yhat):
        assert np.shape(y) == np.shape(yhat)
        return (y - yhat)@(y - yhat).T / len(y) # batch * output

    def backward(self, y, yhat):
        assert np.shape(y) == np.shape(yhat)
        return 2*(yhat - y)          # batch * output 

class CELoss(Loss):

    def forward(self, y, yhat):
        assert(y.shape == yhat.shape)

        return np.log(np.sum(np.exp(yhat), axis=1)+ 1e-100) - np.sum(y * yhat,axis = 1)

    def backward(self, y, yhat):
        assert(y.shape == yhat.shape)

        expo = np.exp(yhat)
        return expo / np.sum(expo, axis=1).reshape((-1,1)) - y

class CELossSoftMax(Loss):
    def forward(self, y, yhat):
        assert(y.shape == yhat.shape)

        return np.sum(-y*yhat, axis = 1)

    def backward(self, y, yhat):
        assert(y.shape == yhat.shape)

        return -1



class BCELoss(Loss):

    def forward (self, y, yhat):
        return -( y * np.maximum( -100, np.log( yhat + 0.01 ) ) + ( 1 - y ) * np.maximum( -100, np.log( 1 - yhat + 0.01 ) ) )

    def backward (self, y, yhat) :
        return - ( ( y / ( yhat + 0.01 ) )- ( ( 1 - y ) / ( 1 - yhat + 0.01 ) ) )





def forward(self, y, yhat):
        assert(y.shape == yhat.shape)
        
        return - (y*np.log(yhat + 1e-100) + (1-y)*np.log(1-yhat+ 1e-100))
    
def backward(self, y, yhat):
        assert(y.shape == yhat.shape)
        
        return ((1-y)/(1-yhat+ 1e-100)) - (y/yhat+ 1e-100)




def forward(self, y, yhat):
    assert(np.shape(y) == np.shape(yhat))

    max1 = np.where( yhat > np.exp(-100) , np.log(yhat), -100)
    max2 = np.where( 1 - np.array(yhat) > np.exp(-100) ,  np.log( 1 - np.array(yhat)), -100)

    return - (y * max1 + ( 1 - y ) * max2)

def backward(self, y, yhat):
    assert(np.shape(y) == np.shape(yhat))

    return - y /(yhat + 1e-100) + (1- y)/(1 - yhat + 1e-100)