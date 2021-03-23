import unittest
import dadi
from adjoint_state_method import ASM_analytic1D, neural_backp_1D
import numpy as np

use_delj_trick = False


class ComputeWeightsTestCase(unittest.TestCase):
    def test_compute_weights(self):
        nu, gamma, h, beta = 2, 0.5, 0.5, 1
        data = dadi.Spectrum.from_file('fs_data.fs')
        ns = data.sample_sizes
        # T = 3
        pts = 19
        xx = dadi.Numerics.default_grid(pts)

        phi_initial = dadi.PhiManip.phi_1D(xx)
        M = dadi.Integration._Mfunc1D(xx, gamma, h)
        MInt = dadi.Integration._Mfunc1D((xx[:-1] + xx[1:]) / 2, gamma, h)
        V = ASM_analytic1D._Vfunc(xx, nu, beta)
        VInt = ASM_analytic1D._Vfunc((xx[:-1] + xx[1:]) / 2, nu, beta=beta)
        dx = np.diff(xx)
        dfactor = dadi.Integration._compute_dfactor(dx)
        delj = dadi.Integration._compute_delj(dx, MInt, VInt)

        dM_dgamma_Int = ASM_analytic1D._Mfunc1D_dgamma((xx[:-1] + xx[1:]) / 2, h)
        dM_dh_Int = ASM_analytic1D._Mfunc1D_dh((xx[:-1] + xx[1:]) / 2, gamma)
        dV_dnu = ASM_analytic1D._Vfunc_dnu(xx, nu, beta)
        dV_dbeta = ASM_analytic1D._Vfunc_dbeta(xx, nu, beta)

        tridiag = ASM_analytic1D.calc_tridiag_matrix(phi_initial, dfactor, MInt, M, V, dx, nu, delj)
        inverse_tridiag = ASM_analytic1D.calc_inverse_tridiag_matrix(tridiag)

        dA_dnu = ASM_analytic1D.calc_dtridiag_dnu(phi_initial, dfactor, dV_dnu, dx, nu, M)
        dA_dbeta = ASM_analytic1D.calc_dtridiag_dbeta(phi_initial, dfactor, dV_dbeta, dx)
        dA_dgamma = ASM_analytic1D.calc_dtridiag_dgamma(phi_initial, dfactor, dM_dgamma_Int, M, dx, delj)
        dA_dh = ASM_analytic1D.calc_dtridiag_dh(phi_initial, dfactor, dM_dh_Int, M, dx, delj)

        dA_dTheta = ASM_analytic1D.get_dtridiag_dtheta(dA_dnu, dA_dbeta, dA_dgamma, dA_dh)
        dA_inverse_dTheta = ASM_analytic1D.get_dtridiag_inverse_dtheta(inverse_tridiag, dA_dTheta)

        timeline_architecture_initial = 0
        timeline_architecture_last = 3
        # nu, gamma, h, beta = 2, 0.5, 0.5, 1
        Theta = [2, 0.5, 0.5, 1, 1]
        adjointer = neural_backp_1D.AdjointStateMethod(timeline_architecture_initial, timeline_architecture_last, ns, pts)
        adjointer.compute_weights(Theta)
        self.assertEqual(phi_initial.all(), adjointer.parameters['phi0'].all())
        self.assertEqual(M.all(), adjointer.parameters['M'].all())
        self.assertEqual(MInt.all(), adjointer.parameters['MInt'].all())
        self.assertEqual(dM_dgamma_Int.all(), adjointer.parameters['dM_dgamma'].all())
        self.assertEqual(dM_dh_Int.all(), adjointer.parameters['dM_dh'].all())
        self.assertEqual(V.all(), adjointer.parameters['V'].all())
        self.assertEqual(VInt.all(), adjointer.parameters['VInt'].all())
        self.assertEqual(dV_dnu.all(), adjointer.parameters['dV_dnu'].all())
        self.assertEqual(dV_dbeta.all(), adjointer.parameters['dV_dbeta'].all())
        self.assertEqual(delj, adjointer.parameters['delj'])
        self.assertEqual(tridiag.all(), adjointer.parameters['A'].all())
        self.assertEqual(inverse_tridiag.all(), adjointer.parameters['A_inv'].all())
        self.assertEqual(dA_dnu.all(), adjointer.parameters['dA_dnu'].all())
        self.assertEqual(dA_dbeta.all(), adjointer.parameters['dA_dbeta'].all())
        self.assertEqual(dA_dgamma.all(), adjointer.parameters['dA_dgamma'].all())
        self.assertEqual(dA_dh.all(), adjointer.parameters['dA_dh'].all())
        self.assertEqual(dA_dTheta.all(), adjointer.parameters['dA_dTheta'].all())
        self.assertEqual(dA_inverse_dTheta.all(), adjointer.parameters['dA_inv_dTheta'].all())

        #self.assertEqual(adjointer.parameters['dA_dTheta'])
        #self.assertEqual(adjointer.parameters['dA_inv_dTheta'])

        """
        phi_dadi = dadi.Integration.one_pop(phi0, adjointer.xx, timeline_architecture_last, nu=Theta[0], gamma=Theta[1],
                                            h=Theta[2], theta0=1, initial_t=0, beta=Theta[3])
        phi, _ = ASM_analytic1D.feedforward(Theta, phi0, adjointer.parameters['A'], adjointer.parameters['A_inv'],
                                            adjointer.xx,
                                            ns, adjointer.dx, adjointer.dfactor, T=timeline_architecture_last, theta0=1,
                                            initial_t=0, delj=adjointer.parameters['delj'])
                                            """
        #phi_backp = list(adjointer.parameters['phi'].values())[-1]
        #self.assertEqual(phi_dadi.all(), phi.all(), phi_backp.all())


suite = unittest.TestLoader().loadTestsFromTestCase(ComputeWeightsTestCase)


if __name__ == '__main__':
    unittest.main()