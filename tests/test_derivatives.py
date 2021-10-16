import unittest
import dadi
from adjoint_state_method import asm_neural_1D, asm_analytic1D
from sympy import *
from scipy.misc import derivative
import numpy as np
import random
import scipy.special
from scipy.integrate import trapz


class DerivativesTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        def partial_derivative(cls, func, var=0, point=[]):
            """var - number of variable to differentiate"""
            args = point[:]

            def wraps(x):
                args[var] = x
                return func(*args)
            return derivative(wraps, point[var], dx=1e-6)
        cls.partial_derivative = partial_derivative

    def test_Vfunc_dnu(self):
        x, nu, beta = symbols('x nu beta')
        x_subs, nu_subs, beta_subs = random.randrange(0, 1), random.randrange(1, 100), random.randrange(1, 10)
        expect_res_simpy = diff(1. / nu * x * (1 - x) * (beta + 1.) ** 2 / (4. * beta), nu).subs({x:x_subs, nu:nu_subs,
                                                                                                  beta:beta_subs})
        analytical_res = asm_neural_1D._Vfunc_dnu(x_subs, nu_subs, beta_subs)
        np.testing.assert_array_equal(expect_res_simpy, analytical_res)
        # self.assertEqual(expect_res_simpy, analytical_res)

    def test_Vfunc_dbeta(self):
        x, nu, beta = symbols('x nu beta')
        x_subs, nu_subs, beta_subs = random.randrange(0, 1), random.randrange(1, 100), random.randrange(1, 10)
        expect_res_simpy = diff(1. / nu * x * (1 - x) * (beta + 1.) ** 2 / (4. * beta), beta).subs({x:x_subs, nu:nu_subs,
                                                                                                  beta:beta_subs})
        analytical_res = asm_neural_1D._Vfunc_dbeta(x_subs, nu_subs, beta_subs)
        np.testing.assert_array_equal(expect_res_simpy, analytical_res)
        # self.assertEqual(expect_res_simpy, analytical_res)

    def test__Mfunc1D_dgamma(self):
        x, gamma, h = symbols('x gamma h')
        x_subs, gamma_subs, h_subs = random.randrange(0, 1), random.randrange(0, 1), random.randrange(0, 1)
        expect_res_simpy = diff(gamma * 2 * x * (h + (1 - 2 * h) * x) * (1 - x), gamma).subs(
            {x: x_subs, gamma: gamma_subs,
             h: h_subs})
        analytical_res = asm_neural_1D._Mfunc1D_dgamma(x_subs, h_subs)
        self.assertEqual(expect_res_simpy, analytical_res)

    def test_Mfunc1D_dh(self):
        x, gamma, h = symbols('x gamma h')
        x_subs, gamma_subs, h_subs = random.randrange(0, 1), random.randrange(0, 1), random.randrange(0, 1)
        expect_res_simpy = diff(gamma * 2 * x * (h + (1 - 2 * h) * x) * (1 - x), h).subs(
            {x: x_subs, gamma: gamma_subs,
             h: h_subs})
        analytical_res = asm_neural_1D._Mfunc1D_dh(x_subs, gamma_subs)
        np.testing.assert_array_equal(expect_res_simpy, analytical_res)
        # self.assertEqual(expect_res_simpy, analytical_res)

    def test_from_phi_1D_direct_dphi_directly(self):
        pts = 19
        xx = np.sort(np.random.random_sample(pts))
        phi = np.sort(np.random.standard_exponential(pts))
        ns = [len(phi) - 1]
        expect_res = self.partial_derivative(asm_analytic1D._from_phi_1D_direct, 0, [phi, ns[0], xx])
        # expect_res = partial_derivative(neural_backp_1D._from_phi_1D_direct, 0, [phi, ns[0], xx])
        direct_derivative_res = asm_neural_1D._from_phi_1D_direct_dphi_directly(ns[0], xx)
        print("expect_res", expect_res)
        print("direct_derivative_res", direct_derivative_res)
        np.testing.assert_array_almost_equal(expect_res, direct_derivative_res, decimal=10)

    def test_dll_dphi(self):
        data = dadi.Spectrum.from_file('fs_data.fs')
        ns = data.sample_sizes  # mask corners
        print("ns", ns)
        pts = 19
        xx = dadi.Numerics.default_grid(pts)
        # xx = np.sort(np.random.random_sample(pts))
        # phi = np.sort(np.random.standard_exponential(pts))
        phi = np.array([0.0191669,  0.02948671, 0.08933759, 0.19649667, 0.24602783, 0.29528289, 0.34076477, 0.46330647,
               0.56886804, 0.61396807, 0.65382797, 0.88665862, 0.93538483, 0.93817732, 1.02521113, 1.16392914,
               1.20318998, 1.62696974, 3.23604856])
        model = dadi.Spectrum.from_phi(phi, [ns[0]], [xx], force_direct=True)
        print(len(data), len(model))
        print("model", model)
        print("data", data)
        # ASM_analytic1D._from_phi_1D_direct(phi, ns[0], xx)
        # ns = [len(phi) - 1]
        expect_res = self.partial_derivative(asm_analytic1D.calc_objective_func, 0, [phi, xx, ns[0], data])
        print("ASM_analytic1D.calc_objective_func", asm_analytic1D.calc_objective_func(phi, xx, ns[0],
                                                                                       data))

        dll_numerical = asm_neural_1D.dll_dphi_numeric(phi, data, ns[0], xx)
        print("dll_numerical", dll_numerical)
        dll_from_phi = asm_neural_1D.dll_dphi_analytical(model, data, ns[0], xx)
        print("dll_from_phi analytical", dll_from_phi)
        ll_dadi = dadi.Inference.ll_multinom(model, data)
        print("ll_dadi", ll_dadi)
        print("expect_res", expect_res)
        dll_dphi = asm_analytic1D.dll_dphi(model, data, ns, xx)
        print("dll_dphi analytic", len(dll_dphi), sum(np.asarray(dll_dphi)))
        dll_dphi_neural = asm_neural_1D.dll_dphi_analytical(model, data, ns[0], xx)
        print("dll_dphi_neural", dll_dphi_neural)
        ll_array_like = np.asarray(dadi.Inference.ll_multinom_per_bin(model, data))
        print("ll_array_like", ll_array_like, type(ll_array_like))
        np.testing.assert_array_almost_equal(expect_res, sum(np.asarray(dll_dphi)), decimal=1)

        # expect_res_neural = neural_backp_1D.dll_dphi(model, data, ns[0], xx)
        # scipy_derivative = self.partial_derivative(ll, 0, [phi, xx, ns[0], data])
        # print(expect_res_neural, "expect_res_neural")
        # print(scipy_derivative, "scipy_derivative")
        # np.testing.assert_array_almost_equal(expect_res_neural, scipy_derivative, decimal=1)

    def test_dmodel_dphi(self):
        def _from_phi_1D_direct(phi, n, xx, mask_corners=True,
                                het_ascertained=None):
            """
            Function from dadi.Spectrum_mod
            Compute sample Spectrum_mod.py from population frequency distribution phi.
            """
            data = np.zeros(n + 1)
            for ii in range(0, n + 1):
                factorx = scipy.special.comb(n, ii) * xx ** ii * (1 - xx) ** (n - ii)
                if het_ascertained == 'xx':
                    factorx *= xx * (1 - xx)
                data[ii] = trapz(factorx * phi, xx)
            return dadi.Spectrum(data, mask_corners=mask_corners)

        data = dadi.Spectrum.from_file('fs_data.fs')
        ns = data.sample_sizes  # mask corners
        print("ns", ns)
        pts = 19
        xx = dadi.Numerics.default_grid(pts)
        # xx = np.sort(np.random.random_sample(pts))
        phi = np.sort(np.random.standard_exponential(pts))
        scipy_derivative = self.partial_derivative(_from_phi_1D_direct, 0, [phi, ns[0], xx])
        dmodel_dphi = asm_neural_1D._from_phi_1D_direct_dphi_directly(ns[0], xx)
        np.testing.assert_array_almost_equal(scipy_derivative, dmodel_dphi, decimal=1)


"""
        data = dadi.Spectrum_mod.py.from_file('fs_data.fs')
        ns = data.sample_sizes
        print("ns", ns)
        nu = 2
        xx = dadi.Numerics.default_grid(ns[0])
        phi = dadi.PhiManip.phi_1D(xx, nu, theta0=1.0, gamma=1, h=0.5, theta=None, beta=1)
        expect_res = partial_derivative(ASM_analyt.calc_model_AFS, 0, [phi, xx, ns])
        analytical_res = ASM_analyt.calc_dmodel_AFS_dphi(xx, ns)
        self.assertEqual(expect_res.any(), analytical_res.any())
        
    def test_from_phi_1D_direct_dphi_analytical(self):
        # test failed 
        def partial_derivative(func, var=0, point=[]):
            args = point[:]

            def wraps(x):
                args[var] = x
                return func(*args)

            return derivative(wraps, point[var], dx=1e-6)

        pts = 10
        xx = np.sort(np.random.random_sample(pts))
        dx = np.diff(xx)
        dfactor = dadi.Integration._compute_dfactor(dx)
        phi = np.sort(np.random.standard_exponential(pts))
        ns = [len(phi) - 1]
        expect_res = partial_derivative(ASM_analytic1D._from_phi_1D_direct, 0, [phi, ns[0], xx])
        analytical_derivative_res = ASM_analytic1D._from_phi_1D_direct_dphi_analytical(ns[0], xx, dfactor)
        self.assertEqual(expect_res.any(), analytical_derivative_res.any())
"""
suite = unittest.TestLoader().loadTestsFromTestCase(DerivativesTestCase)


if __name__ == '__main__':
    unittest.main()
