




class SparseCategoricalCrossEntropySoftmax(nndiy.core.Loss):
    def forward(self, y, yhat):
        self._y = one_hot(y, yhat.shape[1])
        self._yhat = np.where(yhat < MIN_THRESHOLD, MIN_THRESHOLD, yhat)
        self._yhat = np.where(self._yhat > MAX_THRESHOLD, MAX_THRESHOLD, self._yhat)


        self._output = np.log(np.sum(np.exp(self._yhat), axis=1)) \ 
        - np.sum(self._y * self._yhat, axis=1)

    def backward(self):
        _exp = np.exp(self._yhat)
        self._grad_input = _exp / (np.sum(_exp, axis=1).reshape((-1, 1)) + DIVIDE_BY_ZERO_EPS) - self._y