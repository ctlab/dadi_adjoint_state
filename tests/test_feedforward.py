import os
import unittest
import dadi
import numpy as np
from adjoint_state_method import neural_backp_1D
from models import simple_1D_model, Demographics1D
from parameterized import parameterized


class FeedforwardTestCase(unittest.TestCase):
    def test_feedforward_simple1D_model(self):
        nu, gamma, h, beta, theta0 = [2, 0.5, 0.5, 1, 1]
        upper_bound = [100, 1, 1, 10, 1]
        lower_bound = [1e-2, 1e-2, 1e-2, 1e-2, 1]
        data = dadi.Spectrum.from_file('fs_data.fs')
        ns = data.sample_sizes
        T = 3
        pts = 19
        xx = dadi.Numerics.default_grid(pts)
        phi = dadi.PhiManip.phi_1D(xx, nu=nu, theta0=theta0, gamma=gamma,
                                   h=h, theta=None,
                                   beta=beta)

        phi_dadi = dadi.Integration.one_pop(phi, xx, T, nu, gamma=gamma, h=h, theta0=theta0, initial_t=0, beta=beta)
        adjointer = neural_backp_1D.AdjointStateMethod(0, T, ns, pts, xx, upper_bound, lower_bound,
                                                       "simple_1D_model_func", data)
        adjointer.init_model_params([nu, gamma, h, beta, theta0])
        phi = adjointer.forward_propagate()
        print("phi_dadi\n", phi_dadi, "\nphi_feedforward\n", phi)
        np.testing.assert_array_almost_equal(phi_dadi, phi, decimal=1)

    def test_feedforward_two_epoch(self):
        nu, T = 15, 20
        upper_bound = [30, 50]
        lower_bound = [5, 10]
        os.chdir(os.path.dirname(os.path.abspath(Demographics1D.__file__)))
        data = dadi.Spectrum.from_file('fs_data_two_epoch_ASM.fs')
        ns = data.sample_sizes
        T = 20
        pts = 40
        xx = dadi.Numerics.default_grid(pts)
        phi = dadi.PhiManip.phi_1D(xx, nu=nu)

        phi_dadi = dadi.Integration.one_pop(phi, xx, T, nu)
        adjointer = neural_backp_1D.AdjointStateMethod(0, T, ns, pts, xx, upper_bound, lower_bound, "two_epoch_ASM",
                                                       data)
        adjointer.init_model_params([nu, T])
        phi = adjointer.forward_propagate()
        print("phi_dadi\n", phi_dadi, "\nphi_feedforward\n", phi)
        np.testing.assert_array_almost_equal(phi_dadi, phi, decimal=1)


suite = unittest.TestLoader().loadTestsFromTestCase(FeedforwardTestCase)


if __name__ == '__main__':
    unittest.main()