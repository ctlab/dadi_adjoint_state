import unittest
import dadi
from adjoint_state_method import neural_backp_1D
import numpy as np
from models import model_func


class OptimizeParamsTestCase(unittest.TestCase):
    def test_optimize(self):
        popt = [2, 0.5, 0.5, 1, 1]
        upper_bound = [100, 1, 1, 10, 1]
        lower_bound = [1e-2, 1e-2, 1e-2, 1e-2, 1]
        P = np.asarray([popt])  # (training) set of vectors of parameters P
        data = dadi.Spectrum.from_file('fs_data.fs')
        print("len", len(data), data)
        ns = data.sample_sizes
        print("ns", ns)
        pts = 30
        xx = dadi.Numerics.default_grid(pts)
        timeline_architecture_initial = 0
        timeline_architecture_last = 3

        for i in range(0, P.shape[0]):
            nu, gamma, h, beta, theta0 = P[i][0], P[i][1], P[i][2], P[i][3], P[i][4]
            func = model_func.model_func
            # popt_log = dadi.Inference.optimize_log(P[i], data, func, pts,
            #                                       lower_bound=lower_bound,
            #                                       upper_bound=upper_bound,
            #                                       verbose=1)
            # print('Best-fit parameters popt_log: {0}'.format(popt_log))
            popt = dadi.Inference.optimize(P[i], data, func, pts, lower_bound=lower_bound, upper_bound=upper_bound,
                                           verbose=1, flush_delay=0.5, epsilon=1e-3,
                                           gtol=1e-5, multinom=True, maxiter=None, full_output=False,
                                           func_args=[], func_kwargs={}, fixed_params=None, ll_scale=1,
                                           output_file=None)
            print('Best-fit parameters popt: {0}'.format(popt))
            adjointer = neural_backp_1D.NeuralNetwork(timeline_architecture_initial, timeline_architecture_last,
                                                      ns, pts, xx, upper_bound, lower_bound)

            adjointer.fit(P, data, 30, 0.1, 200)
            np.testing.assert_array_equal(np.asarray([1,2]), np.asarray([1,2]))


suite = unittest.TestLoader().loadTestsFromTestCase(OptimizeParamsTestCase)

if __name__ == '__main__':
    unittest.main()
