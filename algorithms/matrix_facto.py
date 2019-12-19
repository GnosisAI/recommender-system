from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from numba import jit
from optimized import _run_epoch

class SVDpp():
    """The *SVD++* algorithm, an extension of :class:`SVD` taking into account
    implicit ratings.
    The prediction :math:`\\hat{r}_{ui}` is set as:
    .. math::
        \hat{r}_{ui} = \mu + b_u + b_i + q_i^T\\left(p_u +
        |I_u|^{-\\frac{1}{2}} \sum_{j \\in I_u}y_j\\right)
        Where the :math:`y_j` terms are a new set of item factors that capture
        implicit ratings. Here, an implicit rating describes the fact that a user
        :math:`u` rated an item :math:`j`, regardless of the rating value.
        If user :math:`u` is unknown, then the bias :math:`b_u` and the factors
        :math:`p_u` are assumed to be zero. The same applies for item :math:`i`
        with :math:`b_i`, :math:`q_i` and :math:`y_i`.
        For details, see section 4 of :cite:`Koren:2008:FMN`. See also
        :cite:`Ricci:2010`, section 5.3.1.
        Just as for :class:`SVD`, the parameters are learned using a SGD on the
        regularized squared error objective.
        Baselines are initialized to ``0``. User and item factors are randomly
        initialized according to a normal distribution, which can be tuned using
        the ``init_mean`` and ``init_std_dev`` parameters.
        You have control over the learning rate :math:`\gamma` and the
        regularization term :math:`\lambda`. Both can be different for each
        kind of parameter (see below). By default, learning rates are set to
        ``0.005`` and regularization terms are set to ``0.02``.
    Args:
        n_factors: The number of factors. Default is ``20``.
        n_epochs: The number of iteration of the SGD procedure. Default is
            ``20``.
        init_mean: The mean of the normal distribution for factor vectors
            initialization. Default is ``0``.
        init_std_dev: The standard deviation of the normal distribution for
            factor vectors initialization. Default is ``0.1``.
        lr_all: The learning rate for all parameters. Default is ``0.007``.
    """

    def __init__(self, n_factors=20, epochs=20, lr=.007, l2=.02):

        self.n_factors = n_factors
        self.n_epochs = epochs
        self.lr = lr
        self.l2 = l2
    
    def fit(self, D):
        self.D = D
        self.sgd(D)

        return self

    def _initialization(self):
        """Initializes biases and latent factor matrixes.
        Args:
            n_user (int): number of different users.
            n_item (int): number of different items.
            n_factors (int): number of factors.
        Returns:
            pu (numpy array): users latent factor matrix.
            qi (numpy array): items latent factor matrix.
            bu (numpy array): users biases vector.
            bi (numpy array): items biases vector.
        """
        self.n_users = self.D.shape[1]
        self.n_items = self.D.shape[1]
        self.global_mean = self.D.mean()

        self.pu = np.random.normal(0, .1, (self.n_users, self.n_factors))
        self.qi = np.random.normal(0, .1, (self.n_items, self.n_factors))

        self.bu = np.zeros(self.n_users)
        self.bi = np.zeros(self.n_items)
        self.yj = np.random.normal(0, .1, (self.n_items, self.n_factors))


    def sgd(self, D):

        self._initialization()
        for current_epoch in range(self.n_epochs):
            print(" processing epoch {}".format(current_epoch))
        # TODO add run epochs
            for u, row in enumerate(D):
                yj, bu, bi, pu, qi = _run_epoch(u, row, self.yj, self.bu, self.bi, self.pu, self.qi, self.lr, self.l2, self.n_factors, self.global_mean)
        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi
        self.yj = yj


    def predict(self, u, i):
        # bui
        rhat = self.global_mean
        rhat += self.bu[u]
        rhat += self.bi[i]

        row = self.D[u]
        Iu = 0.0001
        Iu += len(row[row != 0])  # nb of items rated by u
        u_impl_feedback = sum(self.yj[j] for (j, _) in enumerate(row / np.sqrt(Iu)))
        
        rhat += np.dot(self.qi[i], self.pu[u] + u_impl_feedback)

        return rhat

if __name__ == "__main__":
    import pandas as pd
    from time import time
    d = pd.read_csv("./data/train.csv", index_col='0')
    d = d.drop('Unnamed: 0', axis=1)
    d.columns = range(len(d.columns))
    d = d.fillna(0)
    d = d.iloc[:100, :1000]
    start = time()
    s = SVDpp(epochs=20)
    s.fit(d.values)
    print(s.predict(1, 3))
    print("--- %s seconds ---" % (time() - start))
