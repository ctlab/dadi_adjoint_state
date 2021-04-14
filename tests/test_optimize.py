import logging
import unittest
import dadi
import scipy
from adjoint_state_method import neural_backp_1D
import numpy as np
from models import simple_1D_model, Demographics1D
import time
import os
import autograd

pre_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_file = os.path.join(pre_parent_dir, 'test_optimize.log')


class TestOptimizeParamsBFGS(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # P = np.asarray([popt])  # (training) set of vectors of parameters P
        # print("++++++++++", os.curdir, os.getcwd())
        os.chdir(os.path.dirname(os.path.abspath(Demographics1D.__file__)))
        cls.data_simple_1D = dadi.Spectrum.from_file('fs_data_simple1D.fs')
        cls.data_two_epoch_ASM = dadi.Spectrum.from_file('fs_data_two_epoch_ASM.fs')
        cls.data_three_epoch_ASM = dadi.Spectrum.from_file('fs_data_three_epoch_ASM.fs')
        # ll = dadi.Inference.ll_multinom(cls.data_simple_1D, cls.data_simple_1D)
        # print('**********', ll)
        cls.ns = cls.data_simple_1D.sample_sizes
        cls.pts = 30
        cls.xx = dadi.Numerics.default_grid(cls.pts)
        cls.timeline_architecture_initial = 0
        cls.timeline_architecture_last = 3

    def test_optimize_simple_1D(self):
        childLogger = logging.getLogger("test_optimize_simple_1D")
        childLogger.addHandler(logging.FileHandler(log_file))
        childLogger.setLevel(10)
        P = np.asarray([[2, 0.5, 0.5, 1, 1.]])
        # cls.P = np.asarray([[94.82544756, 0.24847371, 0.56088429, 9.3609537, 1.],
        #                     [0.218, 60, 0.63, 7.09, 1.],
        #                     [59.82827626, 0.2132724, 0.62009149, 7.03617196, 1.],
        #                     [57.55105164, 0.18275621, 0.62845023, 1.89037599, 1.],
        #                     [27.5, 0.4, 0.7, 1.4, 1.]])
        upper_bound = [100, 1, 1, 10, 1]
        lower_bound = [1e-2, 1e-2, 1e-2, 1e-2, 1]
        for i in range(0, P.shape[0]):
            # nu, gamma, h, beta, theta0 = P[i][0], P[i][1], P[i][2], P[i][3], P[i][4]
            func = simple_1D_model.simple_1D_model_func
            childLogger.info('Initial parameters P[{}]={}'.format(i, str(P[i])))

            t1 = time.time()
            popt = dadi.Inference.optimize(P[i], self.data_simple_1D, func, self.pts, lower_bound=lower_bound,
                                           upper_bound=upper_bound,
                                           verbose=1, flush_delay=0.5, epsilon=1e-3,
                                           gtol=1e-5, multinom=True, maxiter=None, full_output=False,
                                           func_args=[], func_kwargs={}, fixed_params=None, ll_scale=1,
                                           output_file=log_file)
            t2 = time.time()
            execution_time = t2 - t1
            childLogger.info('Best-fit parameters popt: {}\nExecution time of dadi.Inference.optimize={}'
                             .format(popt, execution_time))
            np.testing.assert_array_equal(np.asarray([1, 2]), np.asarray([1, 2]))

            t1 = time.time()
            popt_asm = dadi.Inference.optimize_fgrad(P[i], self.data_simple_1D, func, self.pts,
                                                     lower_bound=lower_bound,
                                                     upper_bound=upper_bound,
                                                     verbose=1, flush_delay=0.5,
                                                     gtol=1e-14, multinom=True, maxiter=None, full_output=False,
                                                     func_args=[], func_kwargs={
                    'xx': dadi.Numerics.default_grid,
                    'initial_t': 0,
                    'final_t': 3
                    }, fixed_params=None, ll_scale=1, output_file=log_file)

            t2 = time.time()
            execution_time = t2 - t1
            childLogger.info('Best-fit parameters popt_asm: {}\nExecution time of dadi.Inference.optimize_fgrad={}'
                             .format(popt_asm, execution_time))
            childLogger.info('popt_asm: {}'.format(popt_asm))
            np.testing.assert_array_equal(np.asarray([1, 2]), np.asarray([1, 2]))
            np.testing.assert_array_almost_equal(popt, popt_asm, decimal=1)

    def test_optimize_two_epoch_ASM(self):
        childLogger = logging.getLogger("test_optimize_two_epoch_ASM")
        childLogger.addHandler(logging.FileHandler(log_file))
        childLogger.setLevel(10)
        P = np.asarray([[10]])
        lower_bound = [5]
        upper_bound = [50]
        for i in range(0, P.shape[0]):
            # nu, gamma, h, beta, theta0 = P[i][0], P[i][1], P[i][2], P[i][3], P[i][4]
            func = Demographics1D.two_epoch_ASM
            childLogger.info('Initial parameters P[{}]={}'.format(i, str(P[i])))
            t1 = time.time()
            popt = dadi.Inference.optimize(P[i], self.data_two_epoch_ASM, func, self.pts,
                                           lower_bound=lower_bound,
                                           upper_bound=upper_bound,
                                           verbose=1, flush_delay=0.5, epsilon=1e-3,
                                           gtol=1e-5, multinom=True, maxiter=None, full_output=True,
                                           func_args=[], func_kwargs={
                    'T': self.timeline_architecture_last,
                    'xx': dadi.Numerics.default_grid,
                    'initial_t': 0
                    }, fixed_params=None, ll_scale=1, output_file=log_file)
            t2 = time.time()
            execution_time = t2 - t1
            childLogger.info('Best-fit parameters popt: {}\nExecution time of dadi.Inference.optimize={}'
                             .format(popt, execution_time))

            t1 = time.time()
            popt_asm = dadi.Inference.optimize_fgrad(P[i], self.data_two_epoch_ASM, func, self.pts,
                                                     lower_bound=lower_bound,
                                                     upper_bound=upper_bound,
                                                     verbose=1, flush_delay=0.5,
                                                     gtol=1e-3, multinom=True, maxiter=None, full_output=True,
                                                     func_args=[], func_kwargs={
                    'T': self.timeline_architecture_last,
                    'xx': dadi.Numerics.default_grid,
                    'initial_t': 0,
                    }, fixed_params=None, ll_scale=1, output_file=log_file)

            t2 = time.time()
            execution_time = t2 - t1
            childLogger.info('Best-fit parameters popt_asm: {}\nExecution time of dadi.Inference.optimize_fgrad={}'
                             .format(popt_asm, execution_time))
            childLogger.info('popt_asm: {}'.format(popt_asm))
            np.testing.assert_array_equal(np.asarray([1, 2]), np.asarray([1, 2]))

    def test_optimize_three_epoch_ASM(self):
        childLogger = logging.getLogger("test_optimize_three_epoch_ASM")
        childLogger.addHandler(logging.FileHandler(log_file))
        childLogger.setLevel(10)
        P = np.asarray([[8, 9]])
        lower_bound = [0.01, 0.03]
        upper_bound = [10, 30]
        for i in range(0, P.shape[0]):
            # nu, gamma, h, beta, theta0 = P[i][0], P[i][1], P[i][2], P[i][3], P[i][4]
            func = Demographics1D.three_epoch_ASM
            childLogger.info('Initial parameters P[{}]={}'.format(i, str(P[i])))
            t1 = time.time()
            popt = dadi.Inference.optimize(P[i], self.data_three_epoch_ASM, func, self.pts,
                                           lower_bound=lower_bound,
                                           upper_bound=upper_bound,
                                           verbose=1, flush_delay=0.5, epsilon=1e-3,
                                           gtol=1e-5, multinom=True, maxiter=None, full_output=True,
                                           func_args=[], func_kwargs={
                    'xx': dadi.Numerics.default_grid,
                    'initial_t': 0
                    }, fixed_params=None, ll_scale=1, output_file=log_file)
            t2 = time.time()
            execution_time = t2 - t1
            childLogger.info('Best-fit parameters popt: {}\nExecution time of dadi.Inference.optimize={}'
                             .format(popt, execution_time))

            # t1 = time.time()
            popt_asm = dadi.Inference.optimize_fgrad(P[i], self.data_two_epoch_ASM, func, self.pts,
                                                     lower_bound=lower_bound,
                                                     upper_bound=upper_bound,
                                                     verbose=1, flush_delay=0.5,
                                                     gtol=1e-3, multinom=True, maxiter=None, full_output=True,
                                                     func_args=[], func_kwargs={
                    'T': self.timeline_architecture_last,
                    'xx': dadi.Numerics.default_grid,
                    'initial_t': 0,
                    }, fixed_params=None, ll_scale=1, output_file=log_file)

            # t2 = time.time()
            # execution_time = t2 - t1
            # childLogger.info('Best-fit parameters popt_asm: {}\nExecution time of dadi.Inference.optimize_fgrad={}'
            #                  .format(popt_asm, execution_time))
            # childLogger.info('popt_asm: {}'.format(popt_asm))
            np.testing.assert_array_equal(np.asarray([1, 2]), np.asarray([1, 2]))


class TestOptimizeParamsGradDescent(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # P = np.asarray([popt])  # (training) set of vectors of parameters P
        # print("++++++++++", os.curdir, os.getcwd())
        os.chdir(os.path.dirname(os.path.abspath(Demographics1D.__file__)))
        cls.data_two_epoch_ASM = dadi.Spectrum.from_file('fs_data_two_epoch_ASM.fs')
        cls.data_three_epoch_ASM = dadi.Spectrum.from_file('fs_data_three_epoch_ASM.fs')
        cls.data_simple_1D_model = dadi.Spectrum.from_file('fs_data_simple1D.fs')
        # ll = dadi.Inference.ll_multinom(cls.data_simple_1D, cls.data_simple_1D)
        # print('**********', ll)
        cls.ns = cls.data_simple_1D_model.sample_sizes
        cls.pts = 30
        cls.xx = dadi.Numerics.default_grid(cls.pts)
        cls.timeline_architecture_initial = 0
        cls.timeline_architecture_last = 3
        cls.lr = 0.01
        cls.grad_iter = 1700

    def test_optimize_two_epoch_ASM(self):
        childLogger = logging.getLogger("test_optimize_grad_two_epoch_ASM")
        childLogger.addHandler(logging.FileHandler(log_file))
        childLogger.setLevel(10)
        P = np.asarray([[7]])
        upper_bound = [10]
        lower_bound = [1]
        for i in range(0, P.shape[0]):
            # nu, gamma, h, beta, theta0 = P[i][0], P[i][1], P[i][2], P[i][3], P[i][4]
            func = Demographics1D.two_epoch_ASM
            childLogger.info('Initial parameters P[{}]={}'.format(i, str(P[i])))
            t1 = time.time()
            adjointer = neural_backp_1D.AdjointStateMethod(self.timeline_architecture_initial,
                                                           self.timeline_architecture_last, self.ns, self.pts, self.xx,
                                                           upper_bound, lower_bound, "two_epoch_ASM",
                                                           self.data_two_epoch_ASM)

            popt_asm = adjointer.fit(P, self.data_two_epoch_ASM, self.lr, self.grad_iter)

            t2 = time.time()
            execution_time = t2 - t1
            childLogger.info('Best-fit parameters popt_asm: {}\nExecution time of ASM={}'
                             .format(popt_asm, execution_time))
            childLogger.info('popt_asm: {}'.format(popt_asm))
            np.testing.assert_array_equal(np.asarray([1, 2]), np.asarray([1, 2]))

    def test_optimize_three_epoch_ASM(self):
        childLogger = logging.getLogger("test_optimize_grad_three_epoch_ASM")
        childLogger.addHandler(logging.FileHandler(log_file))
        childLogger.setLevel(10)
        P = np.asarray([[3, 5, 2, 4]])
        upper_bound = [30, 50, 20, 40]
        lower_bound = [0.3, 0.5, 0.2, 0.4]
        for i in range(0, P.shape[0]):
            # nu, gamma, h, beta, theta0 = P[i][0], P[i][1], P[i][2], P[i][3], P[i][4]
            func = Demographics1D.two_epoch_ASM
            childLogger.info('Initial parameters P[{}]={}'.format(i, str(P[i])))
            t1 = time.time()
            adjointer = neural_backp_1D.AdjointStateMethod(self.timeline_architecture_initial,
                                                           self.timeline_architecture_last, self.ns, self.pts, self.xx,
                                                           upper_bound, lower_bound, "three_epoch_ASM",
                                                           self.data_three_epoch_ASM)

            popt_asm = adjointer.fit(P, self.data_three_epoch_ASM, self.lr, self.grad_iter)

            t2 = time.time()
            execution_time = t2 - t1
            childLogger.info('Best-fit parameters popt_asm: {}\nExecution time of ASM={}'
                             .format(popt_asm, execution_time))
            childLogger.info('popt_asm: {}'.format(popt_asm))
            np.testing.assert_array_equal(np.asarray([1, 2]), np.asarray([1, 2]))

    def test_optimize_simple1D_model(self):
        childLogger = logging.getLogger("test_optimize_grad_two_epoch_ASM")
        childLogger.addHandler(logging.FileHandler(log_file))
        childLogger.setLevel(10)
        upper_bound = [100, 1, 1, 10, 1]
        lower_bound = [1e-2, 1e-2, 1e-2, 1e-2, 1]
        P = np.asarray([[100, 0.5001112676618237, 0.49913154400601545, 1.0003446032649062, 1.0]])
        for i in range(0, P.shape[0]):
            # nu, gamma, h, beta, theta0 = P[i][0], P[i][1], P[i][2], P[i][3], P[i][4]
            func = simple_1D_model.simple_1D_model_func
            childLogger.info('Initial parameters P[{}]={}'.format(i, str(P[i])))

            t1 = time.time()
            adjointer = neural_backp_1D.AdjointStateMethod(self.timeline_architecture_initial,
                                                           self.timeline_architecture_last, self.ns, self.pts, self.xx,
                                                           upper_bound, lower_bound, "simple_1D_model_func",
                                                           self.data_simple_1D_model)
            # adjointer.fit(P, self.data_simple_1D_model, self.lr, self.grad_iter)

            g_scipy = dadi.Inference.approx_grad_scipy(P[i], self.data_simple_1D_model, func, self.pts,
                                                       lower_bound=lower_bound, upper_bound=upper_bound,
                                                       verbose=1, flush_delay=0.5,
                                                       gtol=1e-5, multinom=True, maxiter=None, full_output=False,
                                                       func_args=[], func_kwargs={}, fixed_params=None, ll_scale=1,
                                                       output_file=log_file)
            print("+++g_scipy", g_scipy)

            # popt_asm = adjointer.fit(P, self.data_simple_1D_model, self.lr, self.grad_iter)
            #
            # t2 = time.time()
            # execution_time = t2 - t1
            # childLogger.info('Best-fit parameters popt_asm: {}\nExecution time of ASM={}'
            #                  .format(popt_asm, execution_time))
            # childLogger.info('popt_asm: {}'.format(popt_asm))
            # np.testing.assert_array_equal(np.asarray([1, 2]), np.asarray([1, 2]))
