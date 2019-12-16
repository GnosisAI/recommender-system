import unittest
from algorithms.base import  BaseLine
import pandas as pd

class TestALL(unittest.TestCase):

    def test_init_baseline(self):
        print("testing baseline instanciation")
        alg = BaseLine()
        self.assertIsInstance(alg, BaseLine, "creating a baseline estimator failed")

    def test_fit_predict(self): # TODO refacrot this into fit and predict
        df = pd.read_csv("data/train.csv",index_col='userId')
        df = df.reset_index()
        df.columns = range(len(df.columns))

        alg = BaseLine(epochs=10)
        r = alg.fit(df)
        self.assertIsInstance(r, BaseLine, "fitting the baseline failed")
        res = alg.predict(1, 3)
        self.assertLess(res, 5, "rating greater than 5")
        self.assertGreater(res, 0, "negative rating ")


