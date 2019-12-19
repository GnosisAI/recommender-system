from sklearn.base import BaseEstimator
import numpy as np
from .optimized import _run_epoch_base, _get_K
class BaseLine(BaseEstimator):
    def __init__(self, lr=0.003, l2=0.02, epochs=100):
        """"
        Parameters
        ----------
        D : DataFrame
            user movie rating dataframe.
        lr : float
            learning rate.
        l2 : float
           regularisation params .
        epochs: int
            number of training epochs.

        """
        self.lr = lr
        self.l2 = l2
        self.epochs = epochs




    def fit(self, D):
        self.D = D
        self.K = _get_K(D)

        self.mu = self.D.mean()
        self.bi = np.zeros((D.shape[1],))
        self.bu = np.zeros((D.shape[0],))
        # train
        print('start SGD')
        for i in range(self.epochs):
            print(f"epoch {i}")
            _run_epoch_base(self.K, self.mu, self.bu, self.bi, self.lr, self.l2)
        return self

    def predict(self, uid, movid): # TODO support of arrays

        return self.mu + self.bu[uid] + self.bi[movid]