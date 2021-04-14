import unittest
import scipy
from adjoint_state_method import ASM_analytic1D, neural_backp_1D
import numpy as np
import dadi
from dadi_code import Integration
from parameterized import parameterized


class TridiagonalTestCase(unittest.TestCase):
    """
    @parameterized.expand([
        ["foo", "a", "a", ],
        ["bar", "a", "b"],
        ["lee", "b", "b"],
        ])
        """
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
        V = dadi.Integration._Vfunc(xx, nu, beta=beta)
        VInt = dadi.Integration._Vfunc((xx[:-1] + xx[1:]) / 2, nu, beta=beta)

        dx = np.diff(xx)
        dfactor = dadi.Integration._compute_dfactor(dx)
        delj = dadi.Integration._compute_delj(dx, MInt, VInt)
        phi = dadi.PhiManip.phi_1D(xx, nu, theta0=1.0, gamma=1, h=0.5, theta=None, beta=1)
        """ from dadi.Integration._one_pop_const_params"""
        a = np.zeros(phi.shape)
        a[1:] += dfactor[1:] * (-MInt * delj - V[:-1] / (2 * dx))

        c = np.zeros(phi.shape)
        c[:-1] += -dfactor[:-1] * (-MInt * (1 - delj) + V[1:] / (2 * dx))

        b = np.zeros(phi.shape)
        b[:-1] += -dfactor[:-1] * (-MInt * delj - V[:-1] / (2 * dx))
        b[1:] += dfactor[1:] * (-MInt * (1 - delj) + V[1:] / (2 * dx))

        if M[0] <= 0:
            b[0] += (0.5 / nu - M[0]) * 2 / dx[0]
        if M[-1] >= 0:
            b[-1] += -(-0.5 / nu - M[-1]) * 2 / dx[-1]

        tridiag_just_in_place = scipy.sparse.diags([-a[1:], b, -c[:-1]], [-1, 0, 1]).toarray()
        tridiag = ASM_analytic1D.calc_tridiag_matrix(phi, dfactor, MInt, M, V, dx, nu, delj)
        np.testing.assert_equal(tridiag, tridiag_just_in_place)
        timeline_architecture_initial = 0
        timeline_architecture_last = 3
        upper_bound = [100, 1, 1, 10, 1]
        lower_bound = [1e-2, 1e-2, 1e-2, 1e-2, 1]
        adjointer = neural_backp_1D.AdjointStateMethod(timeline_architecture_initial, timeline_architecture_last, [10],
                                                       pts, xx, upper_bound=upper_bound, lower_bound=lower_bound,
                                                       model='simple_1D_model_func', data=data)
        adjointer.init_model_params([nu, gamma, h, beta, theta0])
        adjointer.compute_weights()

        for a_dadi, b_dadi, c_dadi in Integration._one_pop_const_params_check_diags(phi, xx, timeline_architecture_last,
                                                                                    nu=nu, gamma=gamma, h=h,
                                                                                    theta0=theta0, initial_t=0, beta=beta):

            np.testing.assert_equal(a, a_dadi)
            np.testing.assert_equal(b, b_dadi)
            np.testing.assert_equal(c, c_dadi)
            np.testing.assert_equal(a_dadi, adjointer.parameters['a'])
            np.testing.assert_equal(b_dadi, adjointer.parameters['b'])
            np.testing.assert_equal(c_dadi, adjointer.parameters['c'])


suite = unittest.TestLoader().loadTestsFromTestCase(TridiagonalTestCase)


if __name__ == '__main__':
    unittest.main()