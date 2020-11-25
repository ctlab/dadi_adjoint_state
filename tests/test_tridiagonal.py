import unittest
import scipy
from adjoint_state_method import ASM_analytic1D
import numpy as np
import dadi


class TridiagonalTestCase(unittest.TestCase):
    def test_tridiagonal(self):
        nu = 2  # population size
        gamma = 0.5  # Selection coefficient
        h = 0.5  # dominance coefficient
        theta0 = 1
        beta = 1
        pts = 10
        xx = np.sort(np.random.random_sample(pts))
        M = dadi.Integration._Mfunc1D(xx, gamma, h)
        MInt = dadi.Integration._Mfunc1D((xx[:-1] + xx[1:]) / 2, gamma, h)
        V = dadi.Integration._Vfunc(xx, nu, beta)
        VInt = dadi.Integration._Vfunc((xx[:-1] + xx[1:]) / 2, nu, beta=beta)
        dx = np.diff(xx)
        dfactor = dadi.Integration._compute_dfactor(dx)
        delj = dadi.Integration._compute_delj(dx, MInt, VInt)
        #delj = 0.5
        phi = dadi.PhiManip.phi_1D(xx, nu, theta0=1.0, gamma=1, h=0.5, theta=None, beta=1)
        """ from dadi.Integration._one_pop_const_params"""
        a = np.zeros(phi.shape)
        a[1:] += dfactor[1:] * (-MInt * delj - V[:-1] / (2 * dx))

        c = np.zeros(phi.shape)
        c[:-1] += -dfactor[:-1] * (-MInt * (1 - delj) + V[1:] / (2 * dx))

        b = np.zeros(phi.shape)
        b[:-1] += -dfactor[:-1] * (-MInt * delj - V[:-1] / (2 * dx))
        b[1:] += dfactor[1:] * (-MInt * (1 - delj) + V[1:] / (2 * dx))

        if (M[0] <= 0):
            b[0] += (0.5 / nu - M[0]) * 2 / dx[0]
        if (M[-1] >= 0):
            b[-1] += -(-0.5 / nu - M[-1]) * 2 / dx[-1]

        tridiag_expect = scipy.sparse.diags([-a[1:], b, -c[:-1]], [-1, 0, 1]).toarray()
        tridiag = ASM_analytic1D.calc_tridiag_matrix(phi, dfactor, MInt, M, V, dx, nu, delj)
        self.assertEqual(tridiag.all(), tridiag_expect.all())


suite = unittest.TestLoader().loadTestsFromTestCase(TridiagonalTestCase)


if __name__ == '__main__':
    unittest.main()