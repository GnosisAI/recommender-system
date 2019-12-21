from sklearn.base import BaseEstimator
import numpy as np
from  optimized import _get_K, _run_epochs_implicite
from tqdm.auto import tqdm

class CorrelationBasedImplicit(BaseEstimator):


    def __init__(self, l4=0.002, lr=0.005, epochs=10, N=5):
        self.lr = lr
        self.l4 = l4
        self.epochs = epochs
        self.N = N

    def fit(self, D):
        self.D = D
        self.n_users = self.D.shape[0]
        self.n_items = self.D.shape[1]

        self.cij = np.zeros((self.n_items, self.n_items))
        self.wij = np.random.normal(0, .1, (self.n_items, self.n_items))
        self.global_mean = self.D[D > 0].mean()
        self.bi = np.zeros((D.shape[1],))
        self.bu = np.zeros((D.shape[0],))
        self.seen = self.D.astype(bool).astype(int)

        print('start process')
        self.K = _get_K(D)

        _run_epochs_implicite(self.epochs, self.D, self.global_mean, self.seen, self.bu, self.bi, self.wij, self.cij, self.lr, self.l4, self.N)

    def predict(self, u, i):
        Nu = self.seen[u].argsort()[-self.N:][::-1]
        Ru = self.D[u].argsort()[-self.N:][::-1]
        sqrt_Nu = np.sqrt(len(Ru)) + 0.001
        sqrt_Ru = np.sqrt(len(Nu)) + 0.001

        bui = self.global_mean + self.bu[u] + self.bi[i]
        for j in Ru:
            buj = self.global_mean + self.bu[u] + self.bi[j]
            second = (self.K[(u , i)] - buj) * self.wij[u, i]

        second /= sqrt_Ru
        rhat = bui + second
        return rhat


if __name__ == "__main__":
    import pandas as pd
    from time import time
    d = pd.read_csv("./data/train.csv", index_col='0')
    d = d.drop('Unnamed: 0', axis=1)
    d.columns = range(len(d.columns))
    d = d.fillna(0)
    d = d.iloc[:100, :3000]
    start = time()
    s = CorrelationBasedImplicit(epochs=20)
    s.fit(d.values)
    print(s.predict(1, 1))
    print("--- %s seconds ---" % (time() - start))