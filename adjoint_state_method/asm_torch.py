import dadi
import scipy
from scipy.integrate import trapz
from scipy.misc import derivative
import torch
import logging
import os
import numpy as np
from torch import nn

import PhiManip, Inference, Demographics1D
from sklearn.preprocessing import MinMaxScaler

pre_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_file = os.path.join(pre_parent_dir, 'test_torch2.log')
child_logger = logging.getLogger(__name__)
child_logger.addHandler(logging.FileHandler(log_file))
child_logger.setLevel(10)

# phi_file = os.path.join(pre_parent_dir, 'phi.log')
# child_logger = logging.getLogger('phi_log')
# child_logger.addHandler(logging.FileHandler(phi_file))
# child_logger.setLevel(10)


def _from_phi_1D_direct_dphi_directly(n, xx, mask_corners=True,
                                      het_ascertained=None):
    """
    Compute derivative from sample Spectrum_mod.py from population frequency distribution phi.
    """
    data = np.zeros(n + 1)  # for example 20 samples, there are 21 element, - 0 - mutations for 0 samples
    for ii in range(0, n + 1):
        factorx = scipy.special.comb(n, ii) * xx ** ii * (1 - xx) ** (n - ii)
        if het_ascertained == 'xx':
            factorx *= xx * (1 - xx)
        data[ii] = trapz(factorx, xx)
    return dadi.Spectrum(data, mask_corners=mask_corners)


def ll_from_phi(phi, xx, ns, data):
    model = dadi_torch.Spectrum_mod.Spectrum.from_phi(phi, [ns], [xx], force_direct=True)
    return dadi_torch.Inference.ll_multinom(model, data)


def dll_dphi_numeric(phi, data, n, xx):
    """ numerical derivative"""
    scipy_derivative = derivative(ll_from_phi, phi, args=(xx, n, data))
    return scipy_derivative


def scaler(data):
    data = np.asarray([data])
    data = data.transpose()
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    # fit and transform in one step
    normalized = min_max_scaler.fit_transform(data)
    return normalized.transpose().reshape(normalized.shape[0])


class AdjointStateMethod(nn.Module):
    # Parameters are: (nu1F, nu2B, nu2F, m, Tp, T)
    def __init__(self, p0, data, model_func, pts, lower_bound, upper_bound):
        super(AdjointStateMethod, self).__init__()
        self.pts = pts  # n (pts) stores the number of neurons in each layer
        self.ns = data.sample_sizes
        child_logger.info("self.ns={}, type={}, len={}".format(self.ns, type(self.ns), len(self.ns)))
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.model_func = model_func
        self.data = data
        self.p_temp = torch.tensor(p0, dtype=torch.float64)
        self.p_temp.unsqueeze_(1)
        self.p0 = self.p_temp.clone().detach().requires_grad_(True)
        # self.p0 = torch.tensor(p0, dtype=torch.float64, requires_grad=True)  # [nuB, nuF, [nu, gamma, h, beta, Tp, T]
        child_logger.info("self.p0={}, shape={}".format(self.p0, p0.shape))
        self.model = model_func(self.p0.data, self.ns, self.pts)
        self.xx = torch.as_tensor(dadi.Numerics.default_grid(self.pts))
        self.phi_initial = dadi_torch.PhiManip.phi_1D(self.xx)
        if len(self.ns) == 2:
            self.phi_initial = dadi.PhiManip.phi_1D_to_2D(self.xx, self.phi_initial)
        # self.phi = self.phi_initial.clone().detach().requires_grad_(True)
        child_logger.info("self.phi_initial {}={}".format(type(self.phi_initial), self.phi_initial))
        self.ll = torch.tensor(dadi_torch.Inference.ll_multinom(self.model, self.data))
        # self.ll.requires_grad=True
        self.F = torch.zeros(self.phi_initial.shape)  # torch.zeros((1, self.pts,))
        # self.F.requires_grad = True
        self.derivatives = {'dF_dphi': torch.tensor([-1], dtype=torch.float64)}# np.asarray([-1 * np.ones(self.pts)])}

    def forward(self):
        if len(self.ns) == 1:
            self.phi, self.phi_inj = dadi_torch.Integration.one_pop(self.phi_initial, self.xx,
                                                                    self.p0[-1][0], nu=self.p0[0][0])
                                                                             # gamma=self.p0[1],
                                                                             # h=self.p0[2], theta0=torch.tensor(1),
                                                                             # initial_t=self.initial_t, beta=self.p0[3])
        elif len(self.ns) == 2:
            self.phi, self.phi_inj = dadi_torch.Integration.two_pops(self.phi_initial, self.xx, self.p0.data[-1][0],
                                                                     nu1=self.p0.data[0][0], nu2=self.p0.data[1][0],
                                                                     m12=self.p0.data[2][0], m21=self.p0.data[3][0],
                                                                     gamma1=self.p0.data[4][0], gamma2=self.p0.data[5][0],
                                                                     h1=self.p0.data[6][0], h2=self.p0.data[7][0],
                                                                     theta0=self.p0.data[8][0],
                                                                     initial_t=0, frozen1=False,
                                                                     frozen2=False, nomut1=False, nomut2=False, enable_cuda_cached=False)
        child_logger.info("self.phi={}".format(self.phi))
        child_logger.info("self.phi_inj={}".format(self.phi_inj))


    def compute_derivatives_dphi(self):
        """
         Partial derivatives of the cost function with respect to phi and adjoint field:
         adjoint-state lagrange multipliers = adjoint_field
        """
        self.derivatives['dll_dphi'] = dll_dphi_numeric(self.phi, self.data, self.ns[0], self.xx)
        child_logger.info('dll_dphi_numeric={}\n{}'.format(self.derivatives['dll_dphi'].shape, self.derivatives[
                                                                'dll_dphi']))
        child_logger.info('dF_dphi={}\n{}'.format(self.derivatives['dF_dphi'].shape, self.derivatives['dF_dphi']))
        self.derivatives['adjoint_field'] = np.multiply(self.derivatives['dF_dphi'].T,
                                                        np.asarray(self.derivatives['dll_dphi']))
        child_logger.info("adjoint_field={}\n{}".format(self.derivatives['adjoint_field'].shape, self.derivatives[
                                                          'adjoint_field']))

    def compute_derivatives_dTheta(self, lr, eval=1):
        child_logger.info("F before backward={},\n{}".format(self.F.shape, self.F))
        child_logger.info("self.p0.grad before backward={},{}\n{}".format(self.p0, self.p0.data, self.p0.grad))
        child_logger.info("self.F.sum={}, type={}".format(self.F.sum(), type(self.F.sum())))
        self.F.sum().backward(retain_graph=True)
        child_logger.info("F after backward={},\n{}".format(self.F.shape, self.F))
        child_logger.info("self.p0.grad {} after backward={},{}\n{}".format(self.p0.shape, self.p0, self.p0.data,
                                                                            self.p0.grad))
        child_logger.info("self.F.sum={}, type={}".format(self.F.sum(), type(self.F.sum())))
        child_logger.info("self.derivatives['adjoint_field'].resize_(1, 1) shape {}={}".format(self.derivatives[
            'adjoint_field'].resize_(1, 1).shape, self.derivatives[
            'adjoint_field'].resize_(1, 1)))
        self.derivatives['dll_dTheta'] = np.matmul(self.p0.grad, self.derivatives[
            'adjoint_field'].resize_(1, 1))
        child_logger.info("self.derivatives['dll_dTheta']={}".format(self.derivatives['dll_dTheta']))

    def update_parameters(self, lr, iterations):
        i = 0
        grad_approx = dadi_torch.Inference.approx_grad_scipy(np.asarray([self.p0.detach().numpy()[0][0],
                                                                         self.p0.detach().numpy()[1][0]],
                                                                        dtype=object),
                                                             self.data, dadi_torch.Demographics1D.two_epoch,
                                                             self.pts,
                                                             lower_bound=self.lower_bound,
                                                             upper_bound=self.upper_bound,
                                                             verbose=0, flush_delay=0.5, gtol=1e-5, multinom=True,
                                                             maxiter=None, full_output=False,
                                                             func_args=[], func_kwargs={}, fixed_params=None,
                                                             ll_scale=1, output_file=None)
        child_logger.info("grad_approx INIT={}".format(grad_approx))
        child_logger.info("norm2={}".format(np.linalg.norm(self.derivatives['dll_dTheta'])))
        child_logger.info("self.p0.data INIT={}".format(self.p0.data))
        while np.linalg.norm(self.derivatives['dll_dTheta']) > 1e-13 and i < iterations:
            self.p0.data += lr * self.derivatives['dll_dTheta']
            self.compute_derivatives_dTheta(lr, eval=1)
            # self.derivatives['dll_dTheta'] = torch.nn.utils.clip_grad_norm_(self.derivatives['dll_dTheta'],
            #                                                                max_norm=1.0)
            child_logger.info("self.p0.data={}".format(self.p0.data))
            child_logger.info("norm2={}".format(np.linalg.norm(self.derivatives['dll_dTheta'])))
            grad_approx = dadi_torch.Inference.approx_grad_scipy(np.asarray([self.p0.detach().numpy()[0][0],
                                                                             self.p0.detach().numpy()[1][0]],
                                                                            dtype=object),
                                                                 self.data, dadi_torch.Demographics1D.two_epoch,
                                                                 self.pts,
                                                                 lower_bound=self.lower_bound,
                                                                 upper_bound=self.upper_bound,
                                                                 verbose=0, flush_delay=0.5, gtol=1e-5, multinom=True,
                                                                 maxiter=None, full_output=False,
                                                                 func_args=[], func_kwargs={}, fixed_params=None,
                                                                 ll_scale=1, output_file=None)
            child_logger.info("grad_approx={}".format(grad_approx))
            if np.isnan(self.derivatives['dll_dTheta']).any():
                child_logger.info("nan grad")
                return
            i += 1
        child_logger.info("ITER {} GRAD {}\n Best-fit parameters popt: {}".format(i, self.derivatives['dll_dTheta'],
                                                                                  self.p0))

    def fit(self, lr, iterations):
        ll = 0  # stores the ll
        n_c = 0  # stores the number of correct predictions
        failed = 0
        child_logger.info("*********************************\n"
                          "Initial value of parameter's set P={}".format(self.p0))
        child_logger.info("Initial Likelihood={}\nInitial predicted data: {}\n".format(self.ll, self.model))
        with torch.autograd.set_detect_anomaly(True):
            self.forward()
            self.F = self.phi_inj - self.phi
            self.compute_derivatives_dphi()
            self.compute_derivatives_dTheta(lr, eval=0)
            self.update_parameters(lr, iterations)
        check_bounds = [0 if value > upper or value < lower else 1 for (upper, value, lower)
                        in zip(self.upper_bound, self.p0, self.lower_bound)]
        child_logger.info("check bounds={}".format(check_bounds))
        if torch.isnan(self.p0).any() or not all(check_bounds):
            child_logger.info("Optimized params beyond bounds")
            failed += 1

        self.forward()
        self.model = dadi.Spectrum.from_phi(self.phi.detach().numpy(), self.ns, [self.xx.numpy()])
        self.ll = dadi.Inference.ll_multinom(self.model, self.data)

        theta = dadi.Inference.optimal_sfs_scaling(self.model, self.data)
        child_logger.info("data:{}\nmodel_predicted:{}\nLikelihood:{}"
                          "\nValue of scaling theta={}, popt={}".format(self.data, self.model, self.ll, theta, self.p0))
        if np.isclose(self.model * theta, self.data, atol=0.1).all():
            n_c += 1
            child_logger.debug("isclose, n_c={}, with parameters:{}\nLikelihood:{}".
                                format(n_c, self.p0, self.ll))
        return self.p0
