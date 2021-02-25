import logging
import unittest
import dadi
from adjoint_state_method import neural_backp_1D
import numpy as np
from models import model_func
import time
import os

preparentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_file = os.path.join(preparentdir, 'test_optimize.log')
childLogger = logging.getLogger(__name__)
childLogger.addHandler(logging.FileHandler(log_file))
childLogger.setLevel(10)


class OptimizeParamsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # P = np.asarray([[59.82827626, 0.2132724, 0.62009149, 7.03617196, 1.]])
        cls.P = np.asarray([[94.82544756, 0.24847371, 0.56088429, 9.3609537, 1.]])
        # [60, 0.218, 0.63, 7.09, 1.],
        # [59.82827626, 0.2132724, 0.62009149, 7.03617196, 1.],
        # [57.55105164, 0.18275621, 0.62845023, 1.89037599, 1.]])
        # P = np.asarray([[27.5, 0.4, 0.7, 1.4, 1]])
        cls.upper_bound = [100, 1, 1, 10, 1]
        cls.lower_bound = [1e-2, 1e-2, 1e-2, 1e-2, 1]
        # P = np.asarray([popt])  # (training) set of vectors of parameters P
        cls.data = dadi.Spectrum.from_file('fs_data.fs')
        cls.ns = cls.data.sample_sizes
        cls.pts = 30
        cls.xx = dadi.Numerics.default_grid(cls.pts)
        cls.timeline_architecture_initial = 0
        cls.timeline_architecture_last = 3

    def test_optimize(self):
        for i in range(0, self.P.shape[0]):
            # nu, gamma, h, beta, theta0 = P[i][0], P[i][1], P[i][2], P[i][3], P[i][4]
            func = model_func.model_func
            t1 = time.time()
            popt = dadi.Inference.optimize(self.P[i], self.data, func, self.pts, lower_bound=self.lower_bound,
                                           upper_bound=self.upper_bound,
                                           verbose=1, flush_delay=0.5, epsilon=1e-3,
                                           gtol=1e-5, multinom=True, maxiter=None, full_output=False,
                                           func_args=[], func_kwargs={}, fixed_params=None, ll_scale=1,
                                           output_file=log_file)
            t2 = time.time()
            execution_time = t2 - t1
            childLogger.info('Best-fit parameters popt: {}\nExecution time of dadi.Inference.optimize={}'
                             .format(popt, execution_time))
            adjointer = neural_backp_1D.NeuralNetwork(self.timeline_architecture_initial,
                                                      self.timeline_architecture_last,
                                                      self.ns, self.pts, self.xx, self.upper_bound, self.lower_bound)
            execution_time = 0
            for i in range(10):
                t1 = time.time()
                popt_ASM = adjointer.fit(self.P, self.data, 1e7, 500)
                t2 = time.time()
                execution_time += t2 - t1
            execution_time = execution_time / 10
            childLogger.info('Execution time of ASM = {}'.format(execution_time))
            np.testing.assert_array_almost_equal(popt, popt_ASM, decimal=1)


"""
    def test_optimize_log(self):
        for i in range(0, self.P.shape[0]):
            # nu, gamma, h, beta, theta0 = P[i][0], P[i][1], P[i][2], P[i][3], P[i][4]
            func = model_func.model_func
            t1 = time.time()
            popt = dadi.Inference.optimize_log(self.P[i], self.data, func, self.pts, lower_bound=self.lower_bound,
                                               upper_bound=self.upper_bound,
                                               verbose=1, flush_delay=0.5, epsilon=1e-3,
                                               gtol=1e-5, multinom=True, maxiter=None, full_output=False,
                                               func_args=[], func_kwargs={}, fixed_params=None, ll_scale=1,
                                               output_file=log_file)
            t2 = time.time()
            execution_time = t2 - t1
            childLogger.info('Best-fit parameters popt log: {}\nExecution time of dadi.Inference.optimize_log={}'
                             .format(popt, execution_time))
            adjointer = neural_backp_1D.NeuralNetwork(self.timeline_architecture_initial,
                                                      self.timeline_architecture_last,
                                                      self.ns, self.pts, self.xx, self.upper_bound, self.lower_bound)
            execution_time = 0
            for i in range(10):
                # print("param", self.P, "log", np.log(self.P))
                t1 = time.time()
                adjointer.fit(np.log(self.P), self.data, 1e7, 500)
                t2 = time.time()
                execution_time += t2 - t1
            execution_time = execution_time / 10
            childLogger.info('Execution time of ASM = {}'.format(execution_time))

            np.testing.assert_array_equal(np.asarray([1, 2]), np.asarray([1, 2]))
"""

suite = unittest.TestLoader().loadTestsFromTestCase(OptimizeParamsTestCase)

if __name__ == '__main__':
    unittest.main(__name__)
