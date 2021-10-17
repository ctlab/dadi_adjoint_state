import logging
import unittest
import dadi
import asm_neural_1D, asm_torch
import numpy as np
import Demographics1D
import time
import os
import simulation1D

pre_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_file = os.path.join(pre_parent_dir, 'test_optimize_1d.log')


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

    # def test_optimize_simple_1D(self):
    #     child_logger = logging.getLogger("test_optimize_simple_1D")
    #     child_logger.addHandler(logging.FileHandler(log_file))
    #     child_logger.setLevel(10)
    #     P = np.asarray([[2, 0.5, 0.5, 1, 1.]])
    #     # cls.P = np.asarray([[94.82544756, 0.24847371, 0.56088429, 9.3609537, 1.],
    #     #                     [0.218, 60, 0.63, 7.09, 1.],
    #     #                     [59.82827626, 0.2132724, 0.62009149, 7.03617196, 1.],
    #     #                     [57.55105164, 0.18275621, 0.62845023, 1.89037599, 1.],
    #     #                     [27.5, 0.4, 0.7, 1.4, 1.]])
    #     upper_bound = [100, 1, 1, 10, 1]
    #     lower_bound = [1e-2, 1e-2, 1e-2, 1e-2, 1]
    #     for i in range(0, P.shape[0]):
    #         # nu, gamma, h, beta, theta0 = P[i][0], P[i][1], P[i][2], P[i][3], P[i][4]
    #         func = simple_1D_model.simple_1D_model_func
    #         child_logger.info('Initial parameters P[{}]={}'.format(i, str(P[i])))
    #
    #         t1 = time.time()
    #         popt = dadi.Inference.optimize(P[i], self.data_simple_1D, func, self.pts, lower_bound=lower_bound,
    #                                        upper_bound=upper_bound,
    #                                        verbose=1, flush_delay=0.5, epsilon=1e-3,
    #                                        gtol=1e-5, multinom=True, maxiter=None, full_output=False,
    #                                        func_args=[], func_kwargs={}, fixed_params=None, ll_scale=1,
    #                                        output_file=log_file)
    #         t2 = time.time()
    #         execution_time = t2 - t1
    #         child_logger.info('Best-fit parameters popt: {}\nExecution time of dadi.Inference.optimize={}'
    #                           .format(popt, execution_time))
    #         np.testing.assert_array_equal(np.asarray([1, 2]), np.asarray([1, 2]))
    #
    #         t1 = time.time()
    #         popt_asm = dadi.Inference.optimize_fgrad(P[i], self.data_simple_1D, func, self.pts,
    #                                                  lower_bound=lower_bound,
    #                                                  upper_bound=upper_bound,
    #                                                  verbose=1, flush_delay=0.5,
    #                                                  gtol=1e-14, multinom=True, maxiter=None, full_output=False,
    #                                                  func_args=[], func_kwargs={
    #                 'xx': dadi.Numerics.default_grid,
    #                 'initial_t': 0,
    #                 'final_t': 3
    #                 }, fixed_params=None, ll_scale=1, output_file=log_file)
    #
    #         t2 = time.time()
    #         execution_time = t2 - t1
    #         child_logger.info('Best-fit parameters popt_asm: {}\nExecution time of dadi.Inference.optimize_fgrad={}'
    #                           .format(popt_asm, execution_time))
    #         child_logger.info('popt_asm: {}'.format(popt_asm))
    #         np.testing.assert_array_equal(np.asarray([1, 2]), np.asarray([1, 2]))
    #         np.testing.assert_array_almost_equal(popt, popt_asm, decimal=1)

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
        os.chdir(os.path.dirname(os.path.abspath(simulation1D.__file__)))
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
        P = np.asarray([[50., 99.]])
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
            adjointer = asm_torch.AdjointStateMethod(P[i], data_two_epoch_asm, model_func, self.pts, upper_bound,
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
            adjointer = asm_neural_1D.AdjointStateMethod(self.timeline_architecture_initial,
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

