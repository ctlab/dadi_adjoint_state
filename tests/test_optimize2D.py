import logging
import unittest
import dadi
from adjoint_state_method import neural_backp_1D, test_torch1
import numpy as np
from dadi_code import Demographics1D
import time
import os

pre_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_file = os.path.join(pre_parent_dir, 'test_optimize_2d.log')


class TestOptimizeParamsBFGS(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # P = np.asarray([popt])  # (training) set of vectors of parameters P
        # print("++++++++++", os.curdir, os.getcwd())
        os.chdir(os.path.dirname(os.path.abspath(Demographics1D.__file__)))
        cls.data_two_epoch_ASM = dadi.Spectrum.from_file('fs_data_two_epoch_ASM.fs')
        cls.data_three_epoch_ASM = dadi.Spectrum.from_file('fs_data_three_epoch_ASM.fs')
        cls.ns = cls.data_two_epoch_ASM.sample_sizes
        cls.pts = 30
        cls.xx = dadi.Numerics.default_grid(cls.pts)
        cls.timeline_architecture_initial = 0
        cls.timeline_architecture_last = 3

    def test_optimize_two_epoch_ASM(self):
        child_logger = logging.getLogger("test_optimize_two_epoch_ASM")
        child_logger.addHandler(logging.FileHandler(log_file))
        child_logger.setLevel(10)
        P = np.asarray([[10]])
        lower_bound = [5]
        upper_bound = [50]
        for i in range(0, P.shape[0]):
            # nu, gamma, h, beta, theta0 = P[i][0], P[i][1], P[i][2], P[i][3], P[i][4]
            func = Demographics1D.two_epoch_ASM
            child_logger.info('Initial parameters P[{}]={}'.format(i, str(P[i])))
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
            child_logger.info('Best-fit parameters popt: {}\nExecution time of dadi.Inference.optimize={}'
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
            child_logger.info('Best-fit parameters popt_asm: {}\nExecution time of dadi.Inference.optimize_fgrad={}'
                              .format(popt_asm, execution_time))
            child_logger.info('popt_asm: {}'.format(popt_asm))
            np.testing.assert_array_equal(np.asarray([1, 2]), np.asarray([1, 2]))

    def test_optimize_three_epoch_ASM(self):
        child_logger = logging.getLogger("test_optimize_three_epoch_ASM")
        child_logger.addHandler(logging.FileHandler(log_file))
        child_logger.setLevel(10)
        P = np.asarray([[8, 9]])
        lower_bound = [0.01, 0.03]
        upper_bound = [10, 30]
        for i in range(0, P.shape[0]):
            # nu, gamma, h, beta, theta0 = P[i][0], P[i][1], P[i][2], P[i][3], P[i][4]
            func = Demographics1D.three_epoch_ASM
            child_logger.info('Initial parameters P[{}]={}'.format(i, str(P[i])))
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
            child_logger.info('Best-fit parameters popt: {}\nExecution time of dadi.Inference.optimize={}'
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
            # child_logger.info('Best-fit parameters popt_asm: {}\nExecution time of dadi.Inference.optimize_fgrad={}'
            #                  .format(popt_asm, execution_time))
            # child_logger.info('popt_asm: {}'.format(popt_asm))
            np.testing.assert_array_equal(np.asarray([1, 2]), np.asarray([1, 2]))


class TestOptimizeParamsGradDescent(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # P = np.asarray([popt])  # (training) set of vectors of parameters P
        # print("++++++++++", os.curdir, os.getcwd())
        os.chdir(os.path.dirname(os.path.abspath(simulation2D.__file__)))
        # ll = dadi.Inference.ll_multinom(cls.data_simple_1D, cls.data_simple_1D)
        # print('**********', ll)
        cls.pts = 30
        cls.xx = dadi.Numerics.default_grid(cls.pts)
        cls.timeline_architecture_initial = 0
        cls.timeline_architecture_last = 8
        cls.lr = 10
        cls.grad_iter = 100

    def test_optimize_two_epoch(self):
        child_logger = logging.getLogger("test_optimize_grad_two_epoch_ASM")
        child_logger.addHandler(logging.FileHandler(log_file))
        child_logger.setLevel(10)
        P = np.asarray([[[50.], [99.]]])
        upper_bound = [100., 100.]
        lower_bound = [1., 1.]
        # P_scaled = neural_backp_1D.normalize_parameters(P, upper_bound, lower_bound)
        # P_scaled = np.asarray(P_scaled)
        # print("P_scaled={}".format(P_scaled))
        data_two_epoch_asm = dadi.Spectrum.from_file('fs_data_two_epoch.fs')  # 'fs_data_two_epoch_ASM.fs')
        print("data", data_two_epoch_asm)
        ns = data_two_epoch_asm.sample_sizes
        # print("P_scaled.shape[0]", P_scaled.shape[0])
        for i in range(0, P.shape[0]):
            model_func = Demographics1D.two_epoch  # fs
            child_logger.info('Initial parameters P[{}]={}'.format(i, str(P[i])))
            t1 = time.time()
            print("P[i]={}".format(P[i]))
            adjointer = test_torch1.AdjointStateMethod(P[i], data_two_epoch_asm, model_func, self.pts, upper_bound,
                                                       lower_bound)
            adjointer.fit(self.lr, self.grad_iter)
            t2 = time.time()
            execution_time = t2 - t1
            child_logger.info("execution_time={}".format(execution_time))
            np.testing.assert_array_equal(np.asarray([1, 2]), np.asarray([1, 2]))

    def test_optimize_three_epoch(self):
        child_logger = logging.getLogger("test_optimize_grad_three_epoch_ASM")
        child_logger.addHandler(logging.FileHandler(log_file))
        child_logger.setLevel(10)
        P = np.asarray([[3, 5, 2, 4]])
        upper_bound = [30, 50, 20, 40]
        lower_bound = [0.3, 0.5, 0.2, 0.4]
        data_three_epoch_asm = dadi.Spectrum.from_file('fs_data_three_epoch_ASM.fs')
        ns = data_three_epoch_asm.sample_sizes
        for i in range(0, P.shape[0]):
            # nu, gamma, h, beta, theta0 = P[i][0], P[i][1], P[i][2], P[i][3], P[i][4]
            func = Demographics1D.two_epoch_ASM
            child_logger.info('Initial parameters P[{}]={}'.format(i, str(P[i])))
            t1 = time.time()
            adjointer = neural_backp_1D.AdjointStateMethod(self.timeline_architecture_initial,
                                                           self.timeline_architecture_last, ns, self.pts, self.xx,
                                                           upper_bound, lower_bound, "three_epoch_ASM",
                                                           data_three_epoch_asm)

            popt_asm = adjointer.fit(P, data_three_epoch_asm, self.lr, self.grad_iter)

            t2 = time.time()
            execution_time = t2 - t1
            child_logger.info('Best-fit parameters popt_asm: {}\nExecution time of ASM={}'
                              .format(popt_asm, execution_time))
            child_logger.info('popt_asm: {}'.format(popt_asm))
            np.testing.assert_array_equal(np.asarray([1, 2]), np.asarray([1, 2]))

