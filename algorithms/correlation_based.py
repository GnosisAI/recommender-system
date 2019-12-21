from .base import BaseLine
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from operator import itemgetter
import numpy as np

class CorrelationBased(BaseEstimator):


    def __init__(self, l2=100, epochs=10):
        self.l2 = l2
        self.epochs = epochs
        self.baseline = BaseLine(epochs=epochs)

    def calulate_nij(self, i, j):
        users_mv_i = self.D[:, i]
        users_rated_i = set(np.argwhere(users_mv_i != 0).flatten())

        users_mv_j = self.D[:, j]
        users_rated_j = set(np.argwhere(users_mv_j != 0).flatten())

        return len(users_rated_i.intersection(users_rated_j))

    def fit(self, D):
        self.D = D
        print('Calculating similarity')
        self.sim_D = cosine_similarity(np.nan_to_num(D.T, 0)) # side effect nan <= 0
        print('Fiting Baseline Algorithm')
        self.baseline.fit(D)

    def predict(self, usersid, movsid):
        rhats = [] # array for the predictions
        for userid, movid in zip(usersid, movsid):
            user_movies = self.D[userid]
            user_rated_movies = np.argwhere(user_movies != 0).flatten()
            #caluclating nijs
            tmp = {}
            for movj in user_rated_movies:
                tmp[movj] = self.calulate_nij(movid, movj)
            nijs = {k: v/(v+self.l2) for k, v in tmp.items()}
            # get the pij for rated movies
            pijs_rated_movies = self.sim_D[movid, user_rated_movies]
            # calculating sij
            sijs = []
            for i, pij in zip(user_rated_movies, pijs_rated_movies):
                sij = nijs.get(i, 0) * pij
                sijs.append((i, sij))

            # get top 10 similaire movies
            movies_by_sijs = sorted(sijs, key=itemgetter(1), reverse=True)[:10] # TODO add number of knn
            silimarity_normilzer = np.finfo(float).eps
            silimarity_normilzer += sum(map(itemgetter(1), movies_by_sijs))
            # init estimated rating with baseline estimator estimation
            rhat = self.baseline.predict(userid, movid)
            for movj, simj in movies_by_sijs:
                rating = self.D[userid, movj]
                rhat += (simj / silimarity_normilzer) * (rating - self.baseline.predict(userid, movj))
            rhats.append(rhat)
        return np.array(rhats)

if __name__ == "__main__":

    alg = CorrelationBased(epochs=1)
    df = pd.read_csv('data/train.csv', index_col=0)
    df.drop('0', axis=1, inplace=True)

    us, ms = [1, 1], [1, 5891]

    alg.fit(df.values)

    print(alg.predict(us, ms))
