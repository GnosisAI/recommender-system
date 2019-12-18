from sklearn.base import BaseEstimator
import numpy as np
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


    def _get_K(self, D):
        print('Creating Baseline K')
        K = {}
        for userId, row in enumerate(D):
            for movieId, rating in enumerate(row):
                if not np.isnan(rating):
                    K[(userId, movieId)] = rating
        return K

    def fit(self, D):
        self.D_ = D
        self.K = self._get_K(self.D_)

        self.mu = self.D_.mean().mean()
        self.bi = np.zeros((D.shape[1],))
        self.bu = np.zeros((D.shape[0],))
        # train
        for i in range(self.epochs):
            for k, r in self.K.items():
                uid, movid = k
                delta = 2 * (r - self.mu - self.bu[uid]) + 2 * self.l2 * self.bu[uid]
                self.bu[uid] += self.lr * delta

                delta = 2 * (r - self.mu - self.bi[movid]) + 2 * self.l2 * self.bi[movid]
                self.bi[movid] += self.lr * delta
        return self

    def predict(self, uid, movid): # TODO support of arrays

        return self.mu + self.bu[uid] + self.bi[movid]