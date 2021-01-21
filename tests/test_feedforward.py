import unittest
import dadi
from adjoint_state_method import ASM_analytic1D
import numpy as np
from parameterized import parameterized


class FeedforwardTestCase(unittest.TestCase):
    def test_feedforward(self):
        """
        upper_bound = [100, 1, 1, 10, 1]
        lower_bound = [1e-2, 1e-2, 1e-2, 1e-2, 1]
        number_of_traininigs = 10
        # (training) set of vectors of parameters P
        P = list()
        for low, high in zip(lower_bound, upper_bound):
            column = np.random.uniform(low=low, high=high, size=number_of_traininigs)
            P.append(column)

        P = np.asarray(P).T

        data = dadi.Spectrum.from_file('fs_data.fs')
        ns = data.sample_sizes
        pts = 19
        xx = dadi.Numerics.default_grid(pts)
        dx = np.diff(xx)
        phi_initial = dadi.PhiManip.phi_1D(xx)
        timeline_architecture_initial = 0
        timeline_architecture_last = 100

        for i in range(0, P.shape[0]):
"""
        nu, gamma, h, beta, theta0 = 2, 0.5, 0.5, 1, 1
        data = dadi.Spectrum.from_file('fs_data.fs')
        ns = data.sample_sizes
        T = 3
        pts = 19
        xx = dadi.Numerics.default_grid(pts)
        phi = dadi.PhiManip.phi_1D(xx, nu=nu, theta0=theta0, gamma=gamma,
                                   h=h, theta=None,
                                   beta=beta)
        M = dadi.Integration._Mfunc1D(xx, gamma, h)
        MInt = dadi.Integration._Mfunc1D((xx[:-1] + xx[1:]) / 2, gamma, h)
        V = dadi.Integration._Vfunc(xx, nu, beta)
        VInt = dadi.Integration._Vfunc((xx[:-1] + xx[1:]) / 2, nu, beta)
        dx = np.diff(xx)
        dfactor = dadi.Integration._compute_dfactor(dx)
        delj = dadi.Integration._compute_delj(dx, MInt, VInt)
        # tridiag = ASM_analytic1D.calc_tridiag_matrix(phi, dfactor, MInt, M, V, dx, nu, delj)
        # inverse_tridiag = ASM_analytic1D.calc_inverse_tridiag_matrix(tridiag)
        phi_dadi = dadi.Integration.one_pop(phi, xx, T, nu, gamma=gamma, h=h, theta0=theta0, initial_t=0, beta=beta)
        phi, _ = ASM_analytic1D.feedforward([nu, gamma, h, beta, theta0], phi, xx, ns, dx, dfactor, M, MInt, V, 0, T,
                                            delj)
        #self.assertEqual(phi_dadi.all(), phi.all())
        np.testing.assert_array_almost_equal(phi_dadi, phi, decimal=1)
        print("phi_dadi", phi_dadi, "phi_feedforward", phi)


suite = unittest.TestLoader().loadTestsFromTestCase(FeedforwardTestCase)


if __name__ == '__main__':
    unittest.main()