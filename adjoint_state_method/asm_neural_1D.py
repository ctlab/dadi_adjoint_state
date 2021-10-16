import logging
import time
import dadi
import numpy as np
import scipy
import scipy.sparse
import scipy.special
import scipy.stats
from scipy.integrate import trapz
import os
from scipy.misc import derivative
from sklearn.preprocessing import MinMaxScaler
# from tensorflow.python.ops import clip_ops


pre_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_file = os.path.join(pre_parent_dir, 'test_optimize.log')

phi_file = os.path.join(pre_parent_dir, 'phi.log')
phi_Logger = logging.getLogger('phi_log')
phi_Logger.addHandler(logging.FileHandler(phi_file))
phi_Logger.setLevel(10)

child_logger = logging.getLogger(__name__)
child_logger.addHandler(logging.FileHandler(log_file))
child_logger.setLevel(10)

ll_file = os.path.join(pre_parent_dir, 'll.log')
ll_logger = logging.getLogger('ll_log')
ll_logger.addHandler(logging.FileHandler(ll_file))
ll_logger.setLevel(10)

popt_list = list()
popt_values = list()


def st_time(func):
    """
        st decorator to calculate the total time of a func
    """

    def st_func(*args, **key_args):
        t1 = time.time()
        r = func(*args, **key_args)
        t2 = time.time()
        execution_time = t2 - t1
        child_logger.info("Execution time of {}={}".format(func.__name__, execution_time))
        return r

    return st_func


def partial_derivative(func, var=0, point=[]):
    """var - number of variable to differentiate"""
    args = point[:]

    def wraps(x):
        args[var] = x
        return func(*args)

    return derivative(wraps, point[var], dx=1e-6)


def scaler(data):
    data = np.asarray([data])
    data = data.transpose()
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    # fit and transform in one step
    normalized = min_max_scaler.fit_transform(data)
    return normalized.transpose().reshape(normalized.shape[0])


def normalize_parameters(params_list, upper_bounds, lower_bounds):
    params_scaled_list = list()
    for param in params_list:
        params_scaled = list()
        for p, upper_b, lower_b in zip(param, upper_bounds, lower_bounds):
            p_scaled = (p - lower_b) / (upper_b - lower_b)
            params_scaled.append(p_scaled)
        params_scaled_list.append(params_scaled)
    return params_scaled_list


def renoramalize_parameters(pts_scaled, upper_bounds, lower_bounds):
    pts = list()
    for p_scaled, upper, lower in zip(pts_scaled, upper_bounds, lower_bounds):
        p = lower + ((p_scaled - 0) * (upper - lower)) / (1 - 0)
        pts.append(p)
    return pts


def _Vfunc_dnu(x, nu, beta=1):
    return (- x * (1 - x) * (beta + 1.) ** 2) / (nu ** 2 * 4. * beta)


def _Vfunc_dbeta(x, nu, beta=1):
    return (2 * x * (1 - x) * (beta + 1.) * 4 * nu * beta - 4 * nu * x * (1 - x) * (beta + 1) ** 2) / (
            16 * nu ** 2 * beta ** 2)


def _Mfunc1D_dgamma(x, h):
    return 2 * x * (h + (1 - 2 * h) * x) * (1 - x)


def _Mfunc1D_dh(x, gamma):
    return 2 * gamma * x * (2 * x ** 2 - 3 * x + 1)


def calc_tridiag_matrix(phi, dfactor, MInt, M, V, dx, nu, delj):
    a = np.zeros(phi.shape)
    a[1:] += dfactor[1:] * (-MInt * delj - V[:-1] / (2 * dx))
    c = np.zeros(phi.shape)
    c[:-1] += -dfactor[:-1] * (-MInt * (1 - delj) + V[1:] / (2 * dx))
    b = np.zeros(phi.shape)
    b[:-1] += -dfactor[:-1] * (-MInt * delj - V[:-1] / (2 * dx))
    b[1:] += dfactor[1:] * (-MInt * (1 - delj) + V[1:] / (2 * dx))

    if M[0] <= 0:
        b[0] += (0.5 / nu - M[0]) * 2 / dx[0]
    if M[-1] >= 0:
        b[-1] += -(-0.5 / nu - M[-1]) * 2 / dx[-1]

    tridiag_matrix = scipy.sparse.diags([-a[1:], b, -c[:-1]], [-1, 0, 1]).toarray()
    return a, b, c, tridiag_matrix


def calc_dtridiag_dnu(initial_phi, dfactor, dV_dnu, dx, nu, M):
    a = np.zeros(initial_phi.shape)
    a[1:] += -dfactor[1:] * dV_dnu[:-1] / (2 * dx)
    c = np.zeros(initial_phi.shape)
    c[:-1] += -dfactor[:-1] * dV_dnu[1:] / (2 * dx)
    b = np.zeros(initial_phi.shape)
    b[:-1] += dfactor[:-1] * dV_dnu[:-1] / (2 * dx)
    b[1:] += dfactor[1:] * dV_dnu[1:] / (2 * dx)

    if M[0] <= 0:
        b[0] += 1 / (dx[0] * nu ** 2)
    if M[-1] >= 0:
        b[-1] += -1 / (dx[-1] * nu ** 2)

    dtridiag_dnu = scipy.sparse.diags([-a[1:], b, -c[:-1]], [-1, 0, 1]).toarray()
    return dtridiag_dnu


def calc_dtridiag_dbeta(phi_initial, dfactor, dV_dbeta, dx):
    a = np.zeros(phi_initial.shape)
    a[1:] += -dfactor[1:] * dV_dbeta[:-1] / (2 * dx)
    c = np.zeros(phi_initial.shape)
    c[:-1] += -dfactor[:-1] * dV_dbeta[1:] / (2 * dx)
    b = np.zeros(phi_initial.shape)
    b[:-1] += dfactor[:-1] * dV_dbeta[:-1] / (2 * dx)
    b[1:] += dfactor[1:] * dV_dbeta[1:] / (2 * dx)
    # derivative from additional part of b by beta is zero
    dtridiag_dbeta = scipy.sparse.diags([-a[1:], b, -c[:-1]], [-1, 0, 1]).toarray()
    return dtridiag_dbeta


def calc_dtridiag_dgamma(phi_initial, dfactor, dM_dgamma_Int, M, dx, delj):
    a = np.zeros(phi_initial.shape)
    a[1:] += -dfactor[1:] * delj * dM_dgamma_Int
    c = np.zeros(phi_initial.shape)
    c[:-1] += dfactor[:-1] * (1 - delj) * dM_dgamma_Int
    b = np.zeros(phi_initial.shape)
    b[:-1] += dfactor[:-1] * delj * dM_dgamma_Int
    b[1:] += -dfactor[1:] * (1 - delj) * dM_dgamma_Int

    if M[0] <= 0:
        b[0] += -dM_dgamma_Int[0] * 2 / dx[0]
    if M[-1] >= 0:
        b[-1] += dM_dgamma_Int[-1] * 2 / dx[-1]

    dtridiag_dgamma = scipy.sparse.diags([-a[1:], b, -c[:-1]], [-1, 0, 1]).toarray()
    return dtridiag_dgamma


def calc_dtridiag_dh(phi_initial, dfactor, dM_dh, M, dx, delj):
    a = np.zeros(phi_initial.shape)
    a[1:] += -dfactor[1:] * delj * dM_dh
    c = np.zeros(phi_initial.shape)
    c[:-1] += dfactor[:-1] * (1 - delj) * dM_dh
    b = np.zeros(phi_initial.shape)
    b[:-1] += dfactor[:-1] * delj * dM_dh
    b[1:] += -dfactor[1:] * (1 - delj) * dM_dh

    if M[0] <= 0:
        b[0] += -dM_dh[0] * 2 / dx[0]
    if M[-1] >= 0:
        b[-1] += dM_dh[-1] * 2 / dx[-1]

    dtridiag_dh = scipy.sparse.diags([-a[1:], b, -c[:-1]], [-1, 0, 1]).toarray()
    return dtridiag_dh


def calc_tridiag_dtheta0(phi_initial):
    """theta0 is typically constant"""
    a = np.zeros(phi_initial.shape)
    c = np.zeros(phi_initial.shape)
    b = np.zeros(phi_initial.shape)
    dtridiag_dtheta0 = scipy.sparse.diags([-a[1:], b, -c[:-1]], [-1, 0, 1]).toarray()
    return dtridiag_dtheta0


def calc_tridiag_dT(phi_initial):
    a = np.zeros(phi_initial.shape)  # TODO: find time-related M, V functions
    c = np.zeros(phi_initial.shape)
    b = np.zeros(phi_initial.shape)
    dtridiag_dT = scipy.sparse.diags([-a[1:], b, -c[:-1]], [-1, 0, 1]).toarray()
    return dtridiag_dT


def vecnorm(x, ord=2):
    if ord == np.Inf:
        return np.amax(np.abs(x))
    elif ord == -np.Inf:
        return np.amin(np.abs(x))
    else:
        return np.sum(np.abs(x) ** ord, axis=0) ** (1.0 / ord)


def get_dtridiag_inverse_dTheta(inverse_tridiag, dtridiag_dTheta):
    """
    (dA)^(-1)/dTheta = -A^(-1)(dA/dTheta)A^(-1)
    """
    result = np.matmul(-inverse_tridiag, dtridiag_dTheta)
    result = np.matmul(result, inverse_tridiag)
    return result


def calc_injected_and_next_phi(previous_phi, tridiag_matrix, this_dt, xx, theta0):
    injected_phi = dadi.Integration._inject_mutations_1D(previous_phi, this_dt, xx, theta0)
    next_phi = np.matmul(np.linalg.inv(tridiag_matrix), injected_phi)  # TODO: alter in accordance with dadi integration
    #  function and test
    return injected_phi, next_phi


def _from_phi_1D_direct_dphi_directly(n, xx, mask_corners=True,
                                      het_ascertained=None):
    """
    Compute derivative from sample Spectrum_mod.py from population frequency distribution phi.
    """
    # ddx = [0]
    # middle = list(np.diff(xx, n=2))
    # ddx += middle + [0]
    # n = round(n)
    data = np.zeros(n + 1)  # for example 20 samples, there are 21 element, - 0 - mutations for 0 samples
    for ii in range(0, n + 1):
        factorx = scipy.special.comb(n, ii) * xx ** ii * (1 - xx) ** (n - ii)
        if het_ascertained == 'xx':
            factorx *= xx * (1 - xx)
        data[ii] = trapz(factorx, xx)  # 0.5 * factorx, dx
        # data[ii] = 0.5*trapz(factorx, ddx)
    return dadi.Spectrum(data, mask_corners=mask_corners)


def ll_from_phi(phi, xx, ns, data):
    model = dadi.Spectrum.from_phi(phi, [ns], [xx], force_direct=True)
    return dadi.Inference.ll_multinom(model, data)


def dll_dphi_analytical(model, data, n, xx):
    """ analytical derivative"""
    dmodel_dphi = _from_phi_1D_direct_dphi_directly(n, xx)  # TODO: alter method in
    return sum(np.asarray(dmodel_dphi * (data / model - 1)))


def dll_dphi_numeric(phi, data, n, xx):
    """ numerical derivative"""
    scipy_derivative = derivative(ll_from_phi, phi, args=(xx, n, data))
    return scipy_derivative


class AdjointStateMethod(object):
    def __init__(self, time_architecture_initial, time_architecture_last, ns, pts, xx, upper_bound, lower_bound,
                 model_func, data):
        """
        ns: Sample size of resulting Spectrum_mod.py: sequence of P sample sizes for each population.
            For example, (n1,n2): Size of fs to generate.
        pts: Number of grid points to use in integration - number of neurons in each layer
        Number of layers - number of time layers this_dt = min(dt, self.T - current_t) in the forward_propagate func.
        Also there should be:
        phi.ndim == len(ns) == len(xxs), where
        phi: P-dimensional population frequency distribution.
        xxs: Sequence of P one-dimesional grids on which phi is defined.
        """
        self.T = time_architecture_last  # T corresponds to the last layer of the network
        self.initial_t = time_architecture_initial
        self.pts = pts  # n (pts) stores the number of neurons in each layer
        self.ns = ns
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.model_func = model_func
        self.model_name = model_func.__name__
        self.data = data
        # input_size is the number of neurons in the first layer i.e. pts[0]
        # output_size is the number of neurons in the last layer i.e. pts[L]
        # Parameters will store the network parameters,
        # nu - (population size) specifies the relative size of this ancestral population to the
        # reference population. Most often, the reference population is the ancestral, so nu defaults to 1.
        # by default the mutation parameter θ=4Nrefμ is set to 1. If you wish to set a fixed value of θ=4N0μ
        # in your analysis, that information must be provided to the initial ϕ creation function and the
        # Integration functions. When fixing θ, every Integration function must be told what the reference θ is,
        # using the option theta0
        # gamma - selection coefficient, can be < 0?
        # h - dominance coefficient, can be < 0
        # theta0: Proportional to ancestral size. Typically constant.
        # ns - list of sample sizes
        # i.e. the weights (tridiagonal matrix A) and biases (phi_injected)
        # time in dadi models is in genetic units: times are given in units of 2Nref generations - also it is the
        # number of layers
        self.xx = xx
        self.dx = np.diff(self.xx)
        self.dfactor = dadi.Integration._compute_dfactor(self.dx)
        self.parameters = dict()
        self.derivatives = {'dF_dphi': np.asarray([-1 * np.ones(self.pts)])}

    def init_model_params(self, Theta, eval=1):
        global popt_list, popt_values
        if self.model_name == 'simple_1D_model_func':
            popt_list = ['nu', 'gamma', 'h', 'beta', 'theta0']
            self.parameters = {
                'Theta': {
                    'nu': Theta[0], 'gamma': Theta[1], 'h': Theta[2], 'beta': Theta[3], 'theta0': Theta[4]
                    }
                }
        elif self.model_name == 'two_epoch_ASM':  # 'T' is the same as self.T
            popt_list = ['nu']  # popt - parameters for optimization
            self.parameters = {
                'Theta': {
                    'nu': Theta[0],
                    'gamma': 0, 'h': 0.5, 'beta': 1, 'theta0': 1.0
                    }
                }
        elif self.model_name == 'three_epoch_ASM':  # 'T' is the same as self.T
            popt_list = ['nuB', 'nuF']  # popt-number of parameters for optimization [nuB,nuF, TB, TF], TB, TF == const
            popt_values = [Theta[0], Theta[1]]
            if eval == 1:
                self.parameters = {
                    'Theta': {
                        'nuB': Theta[0], 'nuF': Theta[1], 'nu': Theta[0],
                        'gamma': 0, 'h': 0.5, 'beta': 1, 'theta0': 1.0
                        }
                    }
                self.T = Theta[2]  # T = TB
                self.parameters['phi_injected'] = list()
            if eval == 2:
                self.parameters['Theta']['nu'] = Theta[1]
                self.parameters['phi0'] = self.parameters['phi']
                self.T = Theta[3]  # T = TF
                return
        self.parameters['phi_injected'] = list()
        self.parameters['phi0'] = dadi.PhiManip.phi_1D(self.xx, nu=self.parameters['Theta']['nu'],
                                                       theta0=self.parameters['Theta']['theta0'],
                                                       gamma=self.parameters['Theta']['gamma'],
                                                       h=self.parameters['Theta']['h'],
                                                       theta=None,
                                                       beta=self.parameters['Theta']['beta'])
        phi_Logger.info("initial phi:{}, len={}".format(self.parameters['phi0'], len(self.parameters['phi0'])))

    def compute_weights(self, eval=1):
        if eval == 2 and self.model_name == 'three_epoch_ASM':
            self.parameters['V'] = dadi.Integration._Vfunc(self.xx, self.parameters['Theta']['nu'], self.parameters[
                'Theta']['beta'])
            self.parameters['VInt'] = dadi.Integration._Vfunc((self.xx[:-1] + self.xx[1:]) / 2,
                                                              self.parameters['Theta'][
                                                                  'nu'], self.parameters['Theta']['beta'])
            self.parameters['dV_dnu'] = _Vfunc_dnu(self.xx, self.parameters['Theta']['nu'],
                                                   self.parameters['Theta']['beta'])
            self.parameters['dV_dbeta'] = _Vfunc_dbeta(self.xx, self.parameters['Theta']['nu'],
                                                       self.parameters['Theta']['beta'])
            ll_logger.info("dx={}, Mint={}, VInt={}".format(self.dx, self.parameters['MInt'], self.parameters[
                'VInt']))
            self.parameters['delj'] = dadi.Integration._compute_delj(self.dx, self.parameters['MInt'], self.parameters[
                'VInt'])
            self.parameters['dA']['dA_dnuF'] = calc_dtridiag_dnu(self.parameters['phi0'], self.dfactor,
                                                                 self.parameters['dV_dnu'], self.dx,
                                                                 self.parameters['Theta']['nu'], self.parameters['M'])
            self.parameters['a'], self.parameters['b'], self.parameters['c'], \
            self.parameters['A'] = calc_tridiag_matrix(self.parameters['phi0'], self.dfactor, self.parameters['MInt'],
                                                       self.parameters['M'], self.parameters['V'], self.dx,
                                                       self.parameters['Theta']['nu'], self.parameters['delj'])
            self.parameters['A_inv'] = np.linalg.inv(self.parameters['A'])
            dA = list()
            for i in self.parameters['dA'].values():
                dA.append(i)
            self.parameters['dA_dTheta'] = np.asarray(dA)
            self.parameters['dA_inv_dTheta'] = get_dtridiag_inverse_dTheta(self.parameters['A_inv'],
                                                                           self.parameters['dA_dTheta'])
            return
        self.parameters['M'] = dadi.Integration._Mfunc1D(self.xx, self.parameters['Theta']['gamma'], self.parameters[
            'Theta']['h'])
        self.parameters['MInt'] = dadi.Integration._Mfunc1D((self.xx[:-1] + self.xx[1:]) / 2, self.parameters[
            'Theta']['gamma'], self.parameters['Theta']['h'])

        self.parameters['dM_dgamma'] = _Mfunc1D_dgamma((self.xx[:-1] + self.xx[1:]) / 2, self.parameters['Theta'][
            'h'])  # Int
        self.parameters['dM_dh'] = _Mfunc1D_dh((self.xx[:-1] + self.xx[1:]) / 2, self.parameters['Theta']['gamma'])
        # Int

        self.parameters['V'] = dadi.Integration._Vfunc(self.xx, self.parameters['Theta']['nu'], self.parameters[
            'Theta']['beta'])
        self.parameters['VInt'] = dadi.Integration._Vfunc((self.xx[:-1] + self.xx[1:]) / 2, self.parameters['Theta'][
            'nu'], self.parameters['Theta']['beta'])
        self.parameters['dV_dnu'] = _Vfunc_dnu(self.xx, self.parameters['Theta']['nu'],
                                               self.parameters['Theta']['beta'])
        self.parameters['dV_dbeta'] = _Vfunc_dbeta(self.xx, self.parameters['Theta']['nu'],
                                                   self.parameters['Theta']['beta'])
        self.parameters['delj'] = dadi.Integration._compute_delj(self.dx, self.parameters['MInt'], self.parameters[
            'VInt'])

        # Initialize the tridiagonal matrix:
        self.parameters['a'], self.parameters['b'], self.parameters['c'], \
        self.parameters['A'] = calc_tridiag_matrix(self.parameters['phi0'], self.dfactor, self.parameters['MInt'],
                                                   self.parameters['M'], self.parameters['V'], self.dx,
                                                   self.parameters['Theta']['nu'], self.parameters['delj'])
        self.parameters['A_inv'] = np.linalg.inv(self.parameters['A'])
        self.parameters['dA'] = {}
        if self.model_name == 'two_epoch_ASM':
            self.parameters['dA']['dA_dnu'] = calc_dtridiag_dnu(self.parameters['phi0'], self.dfactor,
                                                                self.parameters['dV_dnu'], self.dx,
                                                                self.parameters['Theta']['nu'], self.parameters['M'])
            # self.parameters['dA']['dA_dT'] = calc_tridiag_dT(self.parameters['phi0'])
        if self.model_name == 'three_epoch_ASM':
            self.parameters['dA']['dA_dnuB'] = calc_dtridiag_dnu(self.parameters['phi0'], self.dfactor,
                                                                 self.parameters['dV_dnu'], self.dx,
                                                                 self.parameters['Theta']['nu'], self.parameters['M'])
            return
        elif self.model_name == 'simple_1D_model_func':
            self.parameters['dA']['dA_dnu'] = calc_dtridiag_dnu(self.parameters['phi0'], self.dfactor,
                                                                self.parameters['dV_dnu'], self.dx,
                                                                self.parameters['Theta']['nu'], self.parameters['M'])
            self.parameters['dA']['dA_dbeta'] = calc_dtridiag_dbeta(self.parameters['phi0'], self.dfactor,
                                                                    self.parameters['dV_dbeta'], self.dx)
            self.parameters['dA']['dA_dgamma'] = calc_dtridiag_dgamma(self.parameters['phi0'], self.dfactor,
                                                                      self.parameters['dM_dgamma'],
                                                                      self.parameters['M'],
                                                                      self.dx, self.parameters['delj'])
            self.parameters['dA']['dA_dh'] = calc_dtridiag_dh(self.parameters['phi0'], self.dfactor,
                                                              self.parameters['dM_dh'], self.parameters['M'], self.dx,
                                                              self.parameters['delj'])
            self.parameters['dA']['dA_dtheta0'] = calc_tridiag_dtheta0(self.parameters['phi0'])

        dA = list()
        for i in self.parameters['dA'].values():
            dA.append(i)
        self.parameters['dA_dTheta'] = np.asarray(dA)
        self.parameters['dA_inv_dTheta'] = get_dtridiag_inverse_dTheta(self.parameters['A_inv'],
                                                                       self.parameters['dA_dTheta'])

    def update_weights(self):
        global popt_list
        if 'gamma' in popt_list or 'h' in popt_list:
            self.parameters['M'] = dadi.Integration._Mfunc1D(self.xx, self.parameters['Theta']['gamma'],
                                                             self.parameters['Theta']['h'])
            self.parameters['MInt'] = dadi.Integration._Mfunc1D((self.xx[:-1] + self.xx[1:]) / 2, self.parameters[
                'Theta']['gamma'], self.parameters['Theta']['h'])
        if 'h' in popt_list:
            self.parameters['dM_dgamma'] = _Mfunc1D_dgamma((self.xx[:-1] + self.xx[1:]) / 2, self.parameters['Theta'][
                'h'])  # Int
        if 'gamma' in popt_list:
            self.parameters['dM_dh'] = _Mfunc1D_dh((self.xx[:-1] + self.xx[1:]) / 2, self.parameters['Theta']['gamma'])
        # Int
        if 'nu' in popt_list or 'beta' in popt_list:
            self.parameters['V'] = dadi.Integration._Vfunc(self.xx, self.parameters['Theta']['nu'], self.parameters[
                'Theta']['beta'])
            self.parameters['VInt'] = dadi.Integration._Vfunc((self.xx[:-1] + self.xx[1:]) / 2,  # TODO:remove redundant
                                                              self.parameters['Theta'][
                                                                  'nu'], self.parameters['Theta']['beta'])
            self.parameters['dV_dnu'] = _Vfunc_dnu(self.xx, self.parameters['Theta']['nu'],
                                                   self.parameters['Theta']['beta'])
            self.parameters['dV_dbeta'] = _Vfunc_dbeta(self.xx, self.parameters['Theta']['nu'],
                                                       self.parameters['Theta']['beta'])
        self.parameters['delj'] = dadi.Integration._compute_delj(self.dx, self.parameters['MInt'], self.parameters[
            'VInt'])
        if 'nuB' in popt_list and 'nuF' in popt_list:
            self.parameters['dV_dnuB'] = _Vfunc_dnu(self.xx, self.parameters['Theta']['nuB'],
                                                    self.parameters['Theta']['beta'])
            self.parameters['dV_dnuF'] = _Vfunc_dnu(self.xx, self.parameters['Theta']['nuF'],
                                                    self.parameters['Theta']['beta'])

        if self.model_name == 'two_epoch_ASM':
            self.parameters['dA']['dA_dnu'] = calc_dtridiag_dnu(self.parameters['phi0'], self.dfactor,
                                                                self.parameters['dV_dnu'], self.dx,
                                                                self.parameters['Theta']['nu'], self.parameters['M'])
            # self.parameters['dA']['dA_dT'] = calc_tridiag_dT(self.parameters['phi0'])
        elif self.model_name == 'simple_1D_model_func':
            self.parameters['dA']['dA_dnu'] = calc_dtridiag_dnu(self.parameters['phi0'], self.dfactor,
                                                                self.parameters['dV_dnu'], self.dx,
                                                                self.parameters['Theta']['nu'], self.parameters['M'])
            self.parameters['dA']['dA_dbeta'] = calc_dtridiag_dbeta(self.parameters['phi0'], self.dfactor,
                                                                    self.parameters['dV_dbeta'], self.dx)
            self.parameters['dA']['dA_dgamma'] = calc_dtridiag_dgamma(self.parameters['phi0'], self.dfactor,
                                                                      self.parameters['dM_dgamma'],
                                                                      self.parameters['M'],
                                                                      self.dx, self.parameters['delj'])
            self.parameters['dA']['dA_dh'] = calc_dtridiag_dh(self.parameters['phi0'], self.dfactor,
                                                              self.parameters['dM_dh'], self.parameters['M'], self.dx,
                                                              self.parameters['delj'])
            self.parameters['dA']['dA_dtheta0'] = calc_tridiag_dtheta0(self.parameters['phi0'])
        elif self.model_name == 'three_epoch_ASM':
            self.parameters['dA']['dA_dnuB'] = calc_dtridiag_dnu(self.parameters['phi0'], self.dfactor,
                                                                 self.parameters['dV_dnuB'], self.dx,
                                                                 self.parameters['Theta']['nuB'], self.parameters['M'])
            self.parameters['dA']['dA_dnuF'] = calc_dtridiag_dnu(self.parameters['phi0'], self.dfactor,
                                                                 self.parameters['dV_dnuF'], self.dx,
                                                                 self.parameters['Theta']['nuF'], self.parameters['M'])
        dA = list()
        for i in self.parameters['dA'].values():
            dA.append(i)
        self.parameters['dA_dTheta'] = np.asarray(dA)
        self.parameters['dA_inv_dTheta'] = get_dtridiag_inverse_dTheta(self.parameters['A_inv'],
                                                                       self.parameters['dA_dTheta'])

    def forward_propagate(self):
        # TODO: function from parameters theta that computes self.parameters['phi']
        # take the derivative from that function - backprop
        dt = dadi.Integration._compute_dt(self.dx, self.parameters['Theta']['nu'], [0], self.parameters['Theta'][
            'gamma'], self.parameters['Theta']['h'])
        current_t = self.initial_t
        self.parameters['phi'] = self.parameters['phi0']
        while current_t < self.T:
            this_dt = min(dt, self.T - current_t)
            dadi.Integration._inject_mutations_1D(self.parameters['phi'], this_dt, self.xx,
                                                  self.parameters['Theta']['theta0'])
            self.parameters['phi_injected'].append(self.parameters['phi'])
            r = self.parameters['phi'] / this_dt
            self.parameters['phi'] = dadi.Integration.tridiag.tridiag(self.parameters['a'], self.parameters['b'] + 1
                                                                      / this_dt, self.parameters['c'], r)
            F = self.parameters['phi_injected'] - self.parameters['phi']
            child_logger.info("F functional = {}".format(F))
            current_t += this_dt

        phi_Logger.info("len of forw_phi={}, forw_phi=\n{}".format(len(self.parameters['phi']), self.parameters['phi']))
        return self.parameters['phi']

    def compute_model(self):
        self.parameters['model'] = dadi.Spectrum.from_phi(self.parameters['phi'], self.ns,
                                                          [self.xx], mask_corners=True, force_direct=True)
        phi_Logger.info('model=\n{}'.format(self.parameters['model']))

    def compute_ll(self, data):
        self.parameters['ll'] = dadi.Inference.ll_multinom(self.parameters['model'], data)
        child_logger.info(
            "compute_ll self.parameters['model']={}\nll={}".format(self.parameters['model'], self.parameters['ll']))
        return self.parameters['ll']

    def compute_derivatives_dphi(self, data):
        """
         Partial derivatives of the cost function with respect to phi and adjoint field:
         adjoint-state lagrange multipliers = adjoint_field
        """
        self.derivatives['dll_dphi'] = dll_dphi_numeric(self.parameters['phi'], data, self.ns[0], self.xx)
        phi_Logger.info('dll_dphi_numeric={}'.format(self.derivatives['dll_dphi']))

        self.derivatives['adjoint_field'] = np.multiply(self.derivatives['dF_dphi'].T,
                                                        np.asarray(self.derivatives['dll_dphi']))
        phi_Logger.info("adjoint_field {}={}".format(self.derivatives['adjoint_field'].shape, self.derivatives[
            'adjoint_field']))

    def compute_derivatives_dTheta(self, data, lr, eval=1):
        """ assuming to update weights before call compute_derivatives_dTheta
            eval=0 to compute grad first time
            eval=1 for further calculations """
        self.derivatives['dF_dTheta'] = np.zeros([len(popt_list), self.pts])
        self.derivatives['dll_dTheta'] = np.zeros([len(popt_list), 1])

        for phi_inj in self.parameters['phi_injected']:
            self.derivatives['dF_dTheta'] += np.matmul(self.parameters['dA_inv_dTheta'], phi_inj)
            self.derivatives['dF_dTheta'] = np.matmul(self.derivatives['dF_dTheta'],
                                                      self.parameters['A_inv'])

        phi_Logger.info("self.derivatives['dF_dTheta'] ={}, shape {}".
                        format(self.derivatives['dF_dTheta'], self.derivatives['dF_dTheta'].shape))
        self.derivatives['dll_dTheta'] = np.matmul(self.derivatives['dF_dTheta'], self.derivatives['adjoint_field'])
        return self.derivatives['dll_dTheta']

    def update_parameters(self, lr, iterations, data, theta):
        i = 0
        child_logger.info("GRAD_INIT {}".format(self.derivatives['dll_dTheta']))
        child_logger.info("norm2={}".format(np.linalg.norm(self.derivatives['dll_dTheta'])))
        while np.linalg.norm(self.derivatives['dll_dTheta']) > 1e-13 and i < iterations:
            for index, key in list(enumerate(self.parameters['Theta']))[:len(popt_list)]:
                self.parameters['Theta'][key] += lr * self.derivatives['dll_dTheta'][index][0]
            self.compute_derivatives_dTheta(data, lr, eval=1)
            child_logger.info("norm2={}".format(np.linalg.norm(self.derivatives['dll_dTheta'])))
            if np.isnan(self.derivatives['dll_dTheta']).any():
                child_logger.info("nan grad")
                return
            i += 1
        child_logger.info("ITER {} GRAD {}\n Best-fit parameters popt: {}".format(i, self.derivatives['dll_dTheta'],
                                                                                  self.parameters['Theta']))
        return self.derivatives['dll_dTheta']

    def predict(self):
        self.forward_propagate()
        self.compute_model()
        return self.parameters['model']

    def initialize(self, popt):
        if self.model_name == 'three_epoch_ASM':
            self.init_model_params(popt, eval=1)
            self.compute_weights()
            self.forward_propagate()
            self.init_model_params(popt, eval=2)
            self.compute_weights(eval=2)
        else:
            self.init_model_params(popt)
            self.compute_weights()

    def fit(self, P, data, lr, iterations):
        ll = 0  # stores the ll
        n_c = 0  # stores the number of correct predictions
        failed = 0
        for i in range(0, P.shape[0]):
            child_logger.info("*********************************\n"
                              "Initial value of parameter's set P[{}]={}".format(i, P[i]))
            self.initialize(P[i])
            model_first = self.predict()
            print("model_first", model_first)
            ll = self.compute_ll(data)
            self.compute_derivatives_dphi(data)
            self.compute_derivatives_dTheta(data, lr, eval=0)
            child_logger.info("Initial Likelihood={}\nInitial predicted data: {}\n".format(ll, model_first))
            self.update_parameters(lr, iterations, data, P[i])
            popt = list(self.parameters['Theta'].values())[:len(popt_list)]
            check_bounds = [0 if value > upper or value < lower else 1 for (upper, value, lower)
                            in zip(self.upper_bound, popt,
                                   self.lower_bound)]
            if np.isnan(popt).any() or not all(check_bounds):
                child_logger.info("Optimized params beyond bounds")
                failed += 1
                continue

            self.initialize(popt)
            model_predicted = self.predict()
            ll = self.compute_ll(data)
            theta = dadi.Inference.optimal_sfs_scaling(model_predicted, data)
            child_logger.info("data:{}\nmodel_predicted:{}\nLikelihood:{}"
                              "\nValue of scaling theta={}".format(data, model_predicted, ll, theta))
            if np.isclose(model_predicted * theta, data, atol=0.1).all():
                n_c += 1
                child_logger.debug("isclose, n_c={}, with parameters:{}\nLikelihood:{}".
                                   format(n_c, popt, ll))
        return popt
