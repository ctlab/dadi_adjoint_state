import unittest
import dadi
from adjoint_state_method import ASM_analytic1D, backp_analytic1D
import numpy as np


class ForwardPropagateTestCase(unittest.TestCase):
    def test_forward_prop(self):
        nu, gamma, h, beta = 2, 0.5, 0.5, 1
        data = dadi.Spectrum.from_file('fs_data.fs')
        ns = data.sample_sizes
        pts = 19
        xx = dadi.Numerics.default_grid(pts)
        timeline_architecture_initial = 0
        timeline_architecture_last = 100

        phi_initial = dadi.PhiManip.phi_1D(xx)
        M = dadi.Integration._Mfunc1D(xx, gamma, h)
        MInt = dadi.Integration._Mfunc1D((xx[:-1] + xx[1:]) / 2, gamma, h)
        V = ASM_analytic1D._Vfunc(xx, nu, beta)

        dx = np.diff(xx)
        dfactor = dadi.Integration._compute_dfactor(dx)
        delj = 0.5
        tridiag = ASM_analytic1D.calc_tridiag_matrix(phi_initial, dfactor, MInt, M, V, dx, nu, delj)
        inverse_tridiag = ASM_analytic1D.calc_inverse_tridiag_matrix(tridiag)
        phi_dadi = dadi.Integration.one_pop(phi_initial, xx, timeline_architecture_last, nu, gamma=gamma, h=h, theta0=1,
                                            initial_t=timeline_architecture_initial, beta=beta)
        phi, _ = ASM_analytic1D.feedforward([nu, gamma, h, beta], phi_initial, tridiag, inverse_tridiag, xx, ns, dx, dfactor,
                                            T=timeline_architecture_last, theta0=1,
                                            initial_t=0, delj=delj)

        # nu, gamma, h, beta = 2, 0.5, 0.5, 1
        Theta = [2, 0.5, 0.5, 1, 1]
        adjointer = backp_analytic1D.NeuralNetwork(timeline_architecture_initial, timeline_architecture_last, ns, pts)
        adjointer.forward_propagate(Theta)
        phi_backp = list(adjointer.parameters['phi'].values())[-1]
        self.assertEqual(phi_dadi.all(), phi.all(), phi_backp.all())


suite = unittest.TestLoader().loadTestsFromTestCase(ForwardPropagateTestCase)


if __name__ == '__main__':
    unittest.main()