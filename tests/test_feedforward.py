import unittest
import dadi
from adjoint_state_method import ASM_analyt
import numpy as np


class FeedforwardTestCase(unittest.TestCase):
    def test_feedforward(self):
        nu, gamma, h, beta = 2, 0.5, 0.5, 1
        data = dadi.Spectrum.from_file('fs_data.fs')  # (os.path.join(os.getcwd(), '/adjoint_state_method/fs_data.fs'))
        ns = data.sample_sizes
        T = 20
        pts = 19
        xx = dadi.Numerics.default_grid(pts)
        phi = dadi.PhiManip.phi_1D(xx)
        M = dadi.Integration._Mfunc1D(xx, gamma, h)
        MInt = dadi.Integration._Mfunc1D((xx[:-1] + xx[1:]) / 2, gamma, h)
        V = ASM_analyt._Vfunc(xx, nu, beta)

        dx = np.diff(xx)
        dfactor = dadi.Integration._compute_dfactor(dx)
        delj = 0.5
        tridiag = ASM_analyt.calc_tridiag_matrix(dfactor, MInt, M, V, dx, delj)
        inverse_tridiag = ASM_analyt.calc_inverse_tridiag_matrix(tridiag)
        phi_dadi = dadi.Integration.one_pop(phi, xx, T, nu, gamma=gamma, h=h, theta0=1, initial_t=0, beta=beta)
        phi, _ = ASM_analyt.feedforward([nu, gamma, h, beta], phi, tridiag, inverse_tridiag, xx, ns, dx, dfactor, T=20, theta0=1,
                                initial_t=0, delj=delj)
        self.assertEqual(phi_dadi, phi)



suite = unittest.TestLoader().loadTestsFromTestCase(FeedforwardTestCase)

if __name__ == '__main__':
    unittest.main()