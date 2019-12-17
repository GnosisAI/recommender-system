from base import BaseLine
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from operator import itemgetter
import numpy as np

class CorrelationBased(BaseEstimator):

    def __init__(self, l2=100, epochs=10):
        self.l2 = l2
        self.baseline = BaseLine(epochs=epochs)

    def calculate_nij(self, i, j):
        users_mv_i = self.D.loc[:, i]
        users_rated_i = set(users_mv_i[users_mv_i.notna()].index)
        users_mv_j = self.D.loc[:, j]
        users_rated_j = set(users_mv_j[users_mv_j.notna()].index)
        return len(users_rated_i.intersection(users_rated_j))

    def fit(self, D):
        self.D = D
        print('calculating similarity')
        sim = cosine_similarity(D.fillna(0).values.T)
        self.sim_D = pd.DataFrame(sim, index=D.columns)
        self.sim_D.columns = D.columns
        print('fiting baseline')
        self.baseline.fit(D)

    def predict(self, usersid, movsid):
        rhats = []
        for userid, movid in zip(usersid, movsid):
            user_rated_movies = self.D.loc[userid, :].dropna()
            user_rated_movies_idx = list(user_rated_movies.index)
            # caluclating nij
            tmp = {}
            for movj in user_rated_movies_idx:
                tmp[movj] = self.calculate_nij(movid, movj)
            nijs = {k: v / (v + self.l2) for k, v in tmp.items()}
            # get the pij for rated movies
            pijs_rated_movies = self.sim_D.loc[movid, user_rated_movies_idx].sort_values(ascending=False)
            # calculating sij
            sijs = []
            for i, pij in pijs_rated_movies.items():
                sij = nijs.get(i, 0) * pij
                sijs.append((i, sij))

            # get top 10 similaire movies
            movies_by_sijs = sorted(sijs, key=itemgetter(1), reverse=True)[:10]
            silimarity_normilzer = sum(map(itemgetter(1), movies_by_sijs))

            # init estimated rating with baseline estimator estimation
            rhat = self.baseline.predict(userid, movid)
            for movj, simj in movies_by_sijs:
                rating = self.D.loc[userid, movj]
                rhat += (simj / silimarity_normilzer) * (rating - self.baseline.predict(userid, movj))
            rhats.append(rhat)
        return np.array(rhats)

if __name__ == "__main__":

    alg = CorrelationBased(epochs=1)
    df = pd.read_csv('data/train.csv', index_col=0)
    df.drop('0', axis=1, inplace=True)
    df.columns = df.columns.astype(int)

    us, ms = [1, 1, 2, 2], [1, 2, 1, 2]
    user_rated_movies = df.loc[1, :].dropna()
    user_rated_movies_idx = user_rated_movies.index
    print(list(user_rated_movies_idx))
    alg.fit(df)


    print(alg.predict(us, ms))