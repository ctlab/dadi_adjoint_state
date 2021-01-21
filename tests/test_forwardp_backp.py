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
        dx = np.diff(xx)
        # phi_initial = dadi.PhiManip.phi_1D(xx)
        timeline_architecture_initial = 0
        timeline_architecture_last = 3

        for i in range(0, P.shape[0]):
            # p = P[i].reshape((P[i].size, 1))
            nu, gamma, h, beta, theta0 = P[i][0], P[i][1], P[i][2], P[i][3], P[i][4]

            phi = dadi.PhiManip.phi_1D(xx, nu=nu, theta0=theta0, gamma=gamma,
                                                       h=h, theta=None,
                                                       beta=beta)

            phi_dadi = dadi.Integration.one_pop(phi, xx, timeline_architecture_last, nu=nu, gamma=gamma, h=h,
                                                theta0=theta0, initial_t=timeline_architecture_initial, beta=beta)
            """
            phi, _ = ASM_analytic1D.feedforward([nu, gamma, h, beta], phi_initial, tridiag, inverse_tridiag, xx, ns, dx,
                                            dfactor,
                                            T=timeline_architecture_last, theta0=1,
                                            initial_t=0, delj=delj)
            print(type(phi_dadi[0]), phi_dadi[0], type(phi[0]), phi[0])
            if np.isnan(phi_dadi.all()) or np.isnan(phi.all()):
                print("phi_dadi:", phi_dadi, "phi_feedforward:", phi, "params:", P[i], "\nbreak")
                break
            """
            adjointer = neural_backp_1D.NeuralNetwork(timeline_architecture_initial, timeline_architecture_last,
                                                      ns, pts, xx)
            adjointer.forward_propagate(P[i])
            phi_backp = adjointer.parameters['phi']
            print("phi_backp", phi_backp, "phi_dadi", phi_dadi)
            np.testing.assert_array_almost_equal(phi_backp, phi_dadi, decimal=1)
            """
            for phi_layer_dadi in Integration._one_pop_const_params(phi, xx, timeline_architecture_last,
                                                                    nu=nu, gamma=gamma, h=h, theta0=theta0,
                          initial_t=timeline_architecture_initial, beta=beta):
                for phi_layer_backp in adjointer.forward_propagate(P[i]):

                    np.testing.assert_array_almost_equal(phi_layer_dadi, phi_layer_backp, decimal=1)
                    """
            #adjointer.forward_propagate(P[i])
            #phi_backp = adjointer.parameters['phi'] #.values())[-1]
            #np.testing.assert_array_almost_equal(phi_backp, phi_dadi, decimal=1)
            #self.assertEqual(phi_dadi.all(), phi.all(), phi_backp.all())
            #print("phi_" + str(i), phi_backp)
            #print("params", P[i])


suite = unittest.TestLoader().loadTestsFromTestCase(ForwardPropagateTestCase)


if __name__ == '__main__':
    unittest.main()