import unittest
import dadi
from adjoint_state_method import neural_backp_1D
import numpy as np
from dadi_code import Integration


class ForwardPropagateTestCase(unittest.TestCase):
    def test_forward_prop(self):
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
        timeline_architecture_initial = 0
        timeline_architecture_last = 3

        for i in range(0, P.shape[0]):
            nu, gamma, h, beta, theta0 = P[i][0], P[i][1], P[i][2], P[i][3], P[i][4]

            phi = dadi.PhiManip.phi_1D(xx, nu=nu, theta0=theta0, gamma=gamma,
                                                     h=h, theta=None,
                                                      beta=beta)

            phi_dadi = dadi.Integration.one_pop(phi, xx, timeline_architecture_last, nu=nu, gamma=gamma, h=h,
                                               theta0=theta0, initial_t=timeline_architecture_initial, beta=beta)
            model_dadi = dadi.Spectrum.from_phi(phi_dadi, ns, [xx], force_direct=True)
            ll_dadi = dadi.Inference.ll_multinom(model_dadi, data)

            adjointer = neural_backp_1D.AdjointStateMethod(timeline_architecture_initial, timeline_architecture_last,
                                                           ns, pts, xx)
            adjointer.forward_propagate(P[i])
            adjointer.compute_model()
            adjointer.compute_ll(data)
            ll_backp = adjointer.parameters['ll']
            phi_backp = adjointer.parameters['phi']
            np.testing.assert_array_equal(adjointer.parameters['model'], model_dadi)
            np.testing.assert_array_equal(ll_backp, ll_dadi)
            np.testing.assert_array_equal(phi_backp, phi_dadi)


suite = unittest.TestLoader().loadTestsFromTestCase(ForwardPropagateTestCase)


if __name__ == '__main__':
    unittest.main()