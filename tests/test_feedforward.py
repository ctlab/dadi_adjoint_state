import os
import unittest
import dadi
import numpy as np
from dadi_torch import Integration, Demographics1D
import torch


class FeedforwardTestCase(unittest.TestCase):
    # def test_feedforward_simple1D_model(self):
    #     nu, gamma, h, beta, theta0 = [2, 0.5, 0.5, 1, 1]
    #     upper_bound = [100, 1, 1, 10, 1]
    #     lower_bound = [1e-2, 1e-2, 1e-2, 1e-2, 1]
    #     data = dadi.Spectrum_mod.py.from_file('fs_data.fs')
    #     ns = data.sample_sizes
    #     T = 3
    #     pts = 19
    #     xx = dadi.Numerics.default_grid(pts)
    #     phi = dadi.PhiManip.phi_1D(xx, nu=nu, theta0=theta0, gamma=gamma,
    #                                h=h, theta=None,
    #                                beta=beta)
    #
    #     phi_dadi = dadi.Integration.one_pop(phi, xx, T, nu, gamma=gamma, h=h, theta0=theta0, initial_t=0, beta=beta)
    #     phi = torch.as_tensor(phi)
    #     phi_inj = torch.zeros((1, pts,))
    #     phi_dadi_code, phi_inj = Integration._one_pop_const_params(phi, phi_inj, xx, T, nu=torch.tensor(nu),
    #                                                       gamma=torch.tensor(gamma), h=torch.tensor(h),
    #                                                       theta0=torch.tensor(theta0),
    #                                                       initial_t=0, beta=torch.tensor(beta))
    #     # adjointer = neural_backp_1D.AdjointStateMethod(0, T, ns, pts, xx, upper_bound, lower_bound,
    #     #                                                "simple_1D_model_func", data)
    #     # adjointer.init_model_params([nu, gamma, h, beta, theta0])
    #     # adjointer.compute_weights()
    #     # phi = adjointer.forward_propagate()
    #     print("phi_dadi\n", phi_dadi, "\nphi_dadi_code\n", phi_dadi_code[-1])
    #     np.testing.assert_array_almost_equal(phi_dadi, phi_dadi_code[-1], decimal=1)

    def test_feedforward_two_epoch(self):
        nu, T = 29., 100
        upper_bound = [30., 1000]
        lower_bound = [5., 10]
        os.chdir(os.path.dirname(os.path.abspath(Demographics1D.__file__)))
        data = dadi.Spectrum.from_file('fs_data_two_epoch_ASM.fs')
        ns = data.sample_sizes
        T = 1
        pts = 5
        xx = dadi.Numerics.default_grid(pts)
        phi = dadi.PhiManip.phi_1D(xx, nu=nu)
        print("phi_initial={}".format(phi))
        phi_dadi = dadi.Integration.one_pop(phi, xx, T, nu)
        phi = torch.as_tensor(phi)
        xx = torch.as_tensor(xx)
        T = torch.tensor(T)
        phi_dadi_code, phi_inj = Integration._one_pop_const_params(phi, xx, T, nu=torch.tensor(nu, dtype=torch.float64))
        # adjointer = neural_backp_1D.AdjointStateMethod(0, T, ns, pts, xx, upper_bound, lower_bound,
        #                                                "simple_1D_model_func", data)
        # adjointer.init_model_params([nu, gamma, h, beta, theta0])
        # adjointer.compute_weights()
        # phi = adjointer.forward_propagate()
        print("phi_dadi={}".format(phi_dadi))
        print("phi_dadi_code={}".format(phi_dadi_code))
        np.testing.assert_array_almost_equal(phi_dadi, phi_dadi_code, decimal=1)
        # adjointer = neural_backp_1D.AdjointStateMethod(0, T, ns, pts, xx, upper_bound, lower_bound, "two_epoch_ASM",
        #                                                data)
        # adjointer.init_model_params([nu])
        # adjointer.compute_weights()
        # phi = adjointer.forward_propagate()
        # print("phi_dadi\n", phi_dadi, "\nphi_feedforward\n", phi)
        # np.testing.assert_array_almost_equal(phi_dadi, phi, decimal=1)


suite = unittest.TestLoader().loadTestsFromTestCase(FeedforwardTestCase)

if __name__ == '__main__':
    unittest.main()
