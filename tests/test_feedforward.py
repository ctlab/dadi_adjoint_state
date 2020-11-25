import unittest
import dadi
from adjoint_state_method import ASM_analytic1D
import numpy as np


class FeedforwardTestCase(unittest.TestCase):
    def test_feedforward(self):
        nu, gamma, h, beta = 2, 0.5, 0.5, 1
        data = dadi.Spectrum.from_file('fs_data.fs')
        ns = data.sample_sizes
        T = 3
        pts = 19
        xx = dadi.Numerics.default_grid(pts)
        phi = dadi.PhiManip.phi_1D(xx)
        M = dadi.Integration._Mfunc1D(xx, gamma, h)
        MInt = dadi.Integration._Mfunc1D((xx[:-1] + xx[1:]) / 2, gamma, h)
        V = ASM_analytic1D._Vfunc(xx, nu, beta)

        dx = np.diff(xx)
        dfactor = dadi.Integration._compute_dfactor(dx)
        delj = 0.5
        tridiag = ASM_analytic1D.calc_tridiag_matrix(phi, dfactor, MInt, M, V, dx, nu, delj)
        inverse_tridiag = ASM_analytic1D.calc_inverse_tridiag_matrix(tridiag)
        phi_dadi = dadi.Integration.one_pop(phi, xx, T, nu, gamma=gamma, h=h, theta0=1, initial_t=0, beta=beta)
        phi, _ = ASM_analytic1D.feedforward([nu, gamma, h, beta], phi, tridiag, inverse_tridiag, xx, ns, dx, dfactor,
                                            T=T, theta0=1,
                                            initial_t=0, delj=delj)
        self.assertEqual(phi_dadi.all(), phi.all())


suite = unittest.TestLoader().loadTestsFromTestCase(FeedforwardTestCase)

if __name__ == '__main__':
    unittest.main()