import unittest
from adjoint_state_method import ASM_analyt
from sympy import *
from scipy.misc import derivative
import numpy as np


class ASM_analytTest(unittest.TestCase):

    def test_Vfunc_dnu(self):
        x, nu, beta = symbols('x nu beta')
        expect_res_simpy = diff(1. / nu * x * (1 - x) * (beta + 1.) ** 2 / (4. * beta), nu).subs({x:0.2, nu:1, beta:1})
        analytical_res = ASM_analyt._Vfunc_dnu(0.2, 1, 1)
        self.assertEqual(expect_res_simpy, analytical_res)

    def test_Vfunc_dbeta(self):
        x, nu, beta = symbols('x nu beta')
        expect_res_simpy = diff(1. / nu * x * (1 - x) * (beta + 1.) ** 2 / (4. * beta), beta).subs({x:0.2, nu:1, beta:1})
        analytical_res = ASM_analyt._Vfunc_dbeta(0.2, 1, 1)
        self.assertEqual(expect_res_simpy, analytical_res)

    def test__Mfunc1D_dgamma(self):
        x, gamma, h = symbols('x gamma h')
        expect_res_simpy = diff(gamma * 2 * x * (h + (1 - 2 * h) * x) * (1 - x), gamma).subs(
            {x: 0.2, gamma: 10, h: 5})
        analytical_res = ASM_analyt._Mfunc1D_dgamma(0.2, 5)
        self.assertEqual(expect_res_simpy, analytical_res)

    def test__Mfunc1D_dh(self):
        x, gamma, h = symbols('x gamma h')
        expect_res_simpy = diff(gamma * 2 * x * (h + (1 - 2 * h) * x) * (1 - x), h).subs(
            {x: 0.2, gamma: 10, h: 5})
        analytical_res = ASM_analyt._Mfunc1D_dh(0.2, 10)
        self.assertEqual(expect_res_simpy, analytical_res)

    def test_derivative_model_AFS(self):
        #phi, xx, ns = symbols('phi xx ns')
        def partial_derivative(func, var=0, point=[]):
            args = point[:]
            def wraps(x):
                args[var] = x
                return func(*args)
            return derivative(wraps, point[var], dx=1e-6)

        phi = np.array([0.1, 0.2])
        xx = np.array([0.2, 0.3])
        expect_res = partial_derivative(ASM_analyt.calc_model_AFS, 0, [phi, xx, 2])
        analytical_res = ASM_analyt.calc_dmodel_AFS_dphi(xx, 2)
        self.assertEqual(expect_res.any(), analytical_res.any())



if __name__ == "__main__":
    unittest.main()