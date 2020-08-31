import unittest
from adjoint_state_method import ASM_analyt
from sympy import *
from scipy.misc import derivative
import numpy as np
import dadi


class TridiagonalTestCase(unittest.TestCase):
    def test_tridiagonal(self):
        M = dadi.Integration._Mfunc1D(xx, gamma, h)
        MInt = dadi.Integration._Mfunc1D((xx[:-1] + xx[1:]) / 2, gamma, h)
        V = ASM_analyt._Vfunc(xx, nu, beta)
        dx = np.diff(xx)
        dfactor = dadi.Integration._compute_dfactor(dx)
        delj = 0.5
        tridiag = ASM_analyt.calc_tridiag_matrix(dfactor, MInt, M, V, dx, delj)



suite = unittest.TestLoader().loadTestsFromTestCase(TridiagonalTestCase)

if __name__ == '__main__':
    unittest.main()