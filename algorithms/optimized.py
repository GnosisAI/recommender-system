import numpy as np
from numba import jit
import warnings
@jit(nopython=True)
def _run_epoch_svdpp(D, yj, bu, bi, pu, qi, lr ,l2, n_factors, global_mean):
    for u in range(D.shape[0]):
        row = D[u, :]
        for i, r in enumerate(row):
            # items rated by u. This is COSTLY
            Iu = np.nonzero(row != 0)[0]
            sqrt_Iu = 0.0001 # non zero division
            sqrt_Iu += np.sqrt(len(Iu))

            # compute user implicit feedback
            u_impl_fdb = np.zeros(n_factors, np.double)
            for j in Iu:
                for f in range(n_factors):
                    u_impl_fdb[f] += yj[j, f] / sqrt_Iu


            # compute current error
            dot = 0  # <q_i, (p_u + sum_{j in Iu} y_j / sqrt{Iu}>
            for f in range(n_factors):
                dot += qi[i, f] *(pu[u, f] + u_impl_fdb[f])
    
            err = r - (global_mean + bu[u] + bi[i] + dot)
            # update biases
            bu[u] += lr * (err - l2 * bu[u])
            bi[i] += lr * (err - l2 * bi[i])

            # update factors
            for f in range(n_factors):
                puf = pu[u, f]
                qif = qi[i, f]
                pu[u, f] += lr * (err * qif - l2 * puf)

                qi[i, f] += lr * (err * (puf + u_impl_fdb[f]) -l2 * qif)
                for j in Iu:
                    yj[j, f] += lr * (err * qif / sqrt_Iu - l2 * yj[j, f])

    return  yj, bu, bi, pu, qi


@jit(nopython=True)
def _run_epoch_base(K, mu, bu, bi, lr, l2):
    for k, r in K.items():
                uid, movid = k
                delta = 2 * (r - mu - bu[uid]) + 2 * l2 * bu[uid]
                bu[uid] += lr * delta

                delta = 2 * (r - mu - bi[movid]) + 2 * l2 * bi[movid]
                bi[movid] += lr * delta

@jit(nopython=True)
def _get_K(D):
    K = {}
    for userId in range(D.shape[0]):
        row = D[userId, :]
        for movieId, rating in enumerate(row):
            if not np.isnan(rating):
                K[(userId, movieId)] = rating
    return K