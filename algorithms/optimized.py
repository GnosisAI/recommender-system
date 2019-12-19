import numpy as np
from numba import jit
import warnings
@jit(nopython=True)
def _run_epoch(u, row, yj, bu, bi, pu, qi, lr ,l2, n_factors, global_mean):
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
