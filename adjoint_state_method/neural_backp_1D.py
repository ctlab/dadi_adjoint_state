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

pre_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_file = os.path.join(pre_parent_dir, 'test_optimize.log')
childLogger = logging.getLogger(__name__)
childLogger.addHandler(logging.FileHandler(log_file))
childLogger.setLevel(10)


def st_time(func):
    """
        st decorator to calculate the total time of a func
    """
    def st_func(*args, **key_args):
        t1 = time.time()
        r = func(*args, **key_args)
        t2 = time.time()
        execution_time = t2 - t1
        childLogger.info("Execution time of {}={}".format(func.__name__, execution_time))
        return r

    return st_func


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


def calc_inverse_tridiag_matrix(matrix):
    return np.linalg.inv(matrix)


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


def get_dtridiag_dTheta(dtridiag_dnu, dtridiag_dbeta, dtridiag_dgamma, dtridiag_dh):
    return np.array([dtridiag_dnu, dtridiag_dbeta, dtridiag_dgamma, dtridiag_dh])


def get_dtridiag_inverse_dTheta(inverse_tridiag, dtridiag_dTheta):
    """
    This derivative play role as the activation function
    (dA)^-1/dTheta = -A^(-1)(dA/dTheta)A^(-1)
    """
    result = np.matmul(-inverse_tridiag, dtridiag_dTheta)
    result = np.matmul(result, inverse_tridiag)
    return result


def calc_injected_and_next_phi(previous_phi, tridiag_matrix, this_dt, xx, theta0):
    injected_phi = dadi.Integration._inject_mutations_1D(previous_phi, this_dt, xx, theta0)
    next_phi = np.matmul(np.linalg.inv(tridiag_matrix), injected_phi)
    return injected_phi, next_phi


def _from_phi_1D_direct_dphi_directly(n, xx, mask_corners=True,
                                      het_ascertained=None):
    """
    Compute derivative from sample Spectrum from population frequency distribution phi.
    See from_phi for explanation of arguments.
    """
    n = round(n)
    data = np.zeros(n + 1)  # for example 20 samples, there are 21 element, - 0 - mutations for 0 samples
    for ii in range(0, n + 1):
        factorx = scipy.special.comb(n, ii) * xx ** ii * (1 - xx) ** (n - ii)
        if het_ascertained == 'xx':
            factorx *= xx * (1 - xx)
        data[ii] = trapz(factorx, xx)  # 0.5 * factorx, dx
    return dadi.Spectrum(data, mask_corners=mask_corners)


def dll_dphi(model, data, n, xx):
    """ analytical derivative"""
    dmodel_dphi = _from_phi_1D_direct_dphi_directly(n, xx)
    return sum(np.asarray(dmodel_dphi * (data / model - 1)))


class AdjointStateMethod(object):
    def __init__(self, time_architecture_initial, time_architecture_last, ns, pts, xx, upper_bound,
                 lower_bound):
        """
        ns: Sample size of resulting Spectrum: sequence of P sample sizes for each population.
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
        # ns - list of sample sizes
        # i.e. the weights (tridiagonal matrix A) and biases (phi_injected)
        # time in dadi models is in genetic units: times are given in units of 2Nref generations - also it is the
        # number of layers
        self.xx = xx
        self.dx = np.diff(self.xx)
        self.dfactor = dadi.Integration._compute_dfactor(self.dx)
        self.parameters = dict()

        # Initialize the likelihood:
        self.parameters['ll'] = np.zeros(self.ns[0] + 1)
        self.parameters['model'] = np.zeros(self.ns[0] + 1)

        self.derivatives = {'dF_dphi': np.asarray([-1 * np.ones(self.pts)])}

    def compute_weights(self):

        """
        self.parameters['phi0'] = dadi.PhiManip.phi_1D(self.xx, nu=self.parameters['Theta']['nu'],
                                                       theta0=self.parameters['Theta']['theta0'],
                                                       gamma=self.parameters['Theta']['gamma'],
                                                       h=self.parameters['Theta']['h'],
                                                       theta=None,
                                                       beta=self.parameters['Theta']['beta'])
                                                       """
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
            self.parameters['A'] = calc_tridiag_matrix(self.parameters['phi'], self.dfactor, self.parameters['MInt'],
                                                     self.parameters['M'], self.parameters['V'], self.dx,
                                                     self.parameters['Theta']['nu'], self.parameters['delj'])
        self.parameters['A_inv'] = calc_inverse_tridiag_matrix(self.parameters['A'])
        self.parameters['dA_dnu'] = calc_dtridiag_dnu(self.parameters['phi'], self.dfactor, self.parameters['dV_dnu'],
                                                      self.dx, self.parameters['Theta']['nu'], self.parameters['M'])
        self.parameters['dA_dbeta'] = calc_dtridiag_dbeta(self.parameters['phi'], self.dfactor,
                                                          self.parameters['dV_dbeta'], self.dx)
        self.parameters['dA_dgamma'] = calc_dtridiag_dgamma(self.parameters['phi'], self.dfactor,
                                                            self.parameters['dM_dgamma'], self.parameters['M'],
                                                            self.dx, self.parameters['delj'])
        self.parameters['dA_dh'] = calc_dtridiag_dh(self.parameters['phi'], self.dfactor,
                                                    self.parameters['dM_dh'], self.parameters['M'], self.dx,
                                                    self.parameters['delj'])
        self.parameters['dA_dTheta'] = get_dtridiag_dTheta(self.parameters['dA_dnu'], self.parameters['dA_dbeta'],
                                                           self.parameters['dA_dgamma'], self.parameters['dA_dh'])
        self.parameters['dA_inv_dTheta'] = get_dtridiag_inverse_dTheta(self.parameters['A_inv'],
                                                                       self.parameters['dA_dTheta'])
        """
        self.derivatives = {'dF_dtheta': np.zeros([self.parameters['dA_inv_dtheta'].shape[0], self.ns[0]],
                                                  dtype=float),
                            'dll_dtheta': ...,
                            'dF_dphi': -1 * np.ones(ns),
                            'dll_dphi': np.zeros(self.ns[0] + 1),
                            'adjoint_field': np.zeros(self.ns[0])}
        """

    def forward_propagate(self, Theta):
        self.parameters = {
            'Theta': {
                'nu': Theta[0], 'gamma': Theta[1], 'h': Theta[2], 'beta': Theta[3], 'theta0': Theta[4]
                }
            }
        """
        self.parameters['phi0'] = dadi.PhiManip.phi_1D(self.xx, nu=self.parameters['Theta']['nu'],
                                                       theta0=self.parameters['Theta']['theta0'],
                                                       gamma=self.parameters['Theta']['gamma'],
                                                       h=self.parameters['Theta']['h'],
                                                       theta=None,
                                                       beta=self.parameters['Theta']['beta'])
                                                       """

        self.parameters['phi_injected'] = list()  # self.parameters['phi0']
        # self.parameters['phi'] = dict()  # self.parameters['phi0']
        self.parameters['phi'] = dadi.PhiManip.phi_1D(self.xx, nu=self.parameters['Theta']['nu'],
                                                      theta0=self.parameters['Theta']['theta0'],
                                                      gamma=self.parameters['Theta']['gamma'],
                                                      h=self.parameters['Theta']['h'],
                                                      theta=None,
                                                      beta=self.parameters['Theta']['beta'])

        self.compute_weights()
        # Calculate the activations (phi_injected) and output (phi) for every layer t
        dt = dadi.Integration._compute_dt(self.dx, self.parameters['Theta']['nu'], [0], self.parameters['Theta'][
            'gamma'], self.parameters['Theta']['h'])
        current_t = self.initial_t
        # previous_phi = self.parameters['phi0']
        while current_t < self.T:
            this_dt = min(dt, self.T - current_t)
            self.parameters['phi'] = dadi.Integration._inject_mutations_1D(self.parameters['phi'], this_dt, self.xx,
                                                                           self.parameters['Theta']['theta0'])
            if abs(round(self.T - current_t) - (self.T - current_t)) < 1e-14:  # phi_inj is only needed where t=T
                self.parameters['phi_injected'].append(self.parameters['phi'])  # to do: test for it
            r = self.parameters['phi'] / this_dt
            self.parameters['phi'] = dadi.Integration.tridiag.tridiag(self.parameters['a'], self.parameters['b'] + 1
                                                                      / this_dt, self.parameters['c'], r)
            current_t += this_dt

    def compute_model(self):
        self.parameters['model'] = dadi.Spectrum.from_phi(self.parameters['phi'], self.ns,
                                                          [self.xx], force_direct=True)

    def compute_ll(self, data):
        self.parameters['ll'] = dadi.Inference.ll_multinom(self.parameters['model'], data)
        return self.parameters['ll']
        # self.parameters['ll'] = data * np.log(self.parameters['model']) - self.parameters['model'] - np.log(data)

    def compute_derivatives_dphi(self, data):
        """
         Partial derivatives of the cost function with respect to phi + adjoint field:
        """
        self.derivatives['dll_dphi'] = dll_dphi(self.parameters['model'], data, self.ns[0], self.dx)
        # adjoint-state lagrange multipliers - adjoint_field
        self.derivatives['adjoint_field'] = np.multiply(self.derivatives['dF_dphi'].T,
                                                        np.asarray(self.derivatives['dll_dphi']))

    def compute_derivatives_dTheta(self):
        """ assuming to update weights before call compute_derivatives_dTheta """
        self.derivatives['dF_dTheta'] = 0
        self.derivatives['dll_dTheta'] = 0
        for phi_inj in self.parameters['phi_injected'][:-1]:
            self.derivatives['dF_dTheta'] += np.matmul(self.parameters['dA_inv_dTheta'], phi_inj)
            self.derivatives['dF_dTheta'] = np.matmul(self.derivatives['dF_dTheta'],
                                                      self.parameters['A_inv'])
        self.derivatives['dF_dTheta'] += np.matmul(self.parameters['dA_inv_dTheta'], self.parameters['phi_injected'][
            -1])
        self.derivatives['dll_dTheta'] = np.matmul(self.derivatives['dF_dTheta'], self.derivatives['adjoint_field'])
        return self.derivatives['dll_dTheta']

    @st_time
    def update_parameters(self, lr, iterations):
        # lr, grad_iter = 0.1, 500
        i = 0
        while np.any(np.ndarray.tolist((self.derivatives['dll_dTheta'] > 1e-20))) and i < iterations:
            #  np.linalg.norm(self.derivatives['dll_dTheta'], ord=1) > 1e-30
            for index, key in list(enumerate(self.parameters['Theta']))[:-1]:
                self.parameters['Theta'][key] += lr * self.derivatives['dll_dTheta'][index][0]
            self.compute_weights()
            self.compute_derivatives_dTheta()
            if np.isnan(self.derivatives['dll_dTheta']).any():
                childLogger.info("nan grad")
                return

            # Just for check convergence
            #     self.forward_propagate(list(self.parameters['Theta'].values()))
            #     self.compute_model()
            #     self.compute_ll(data)
            #     if i == 0 or i % 100 == 0:
            #         childLogger.info("{}   , {}   ,   {}".format(i, self.parameters['ll'], self.parameters['Theta']))
            i += 1
        childLogger.info("Best-fit parameters popt: {}".format(self.parameters['Theta']))
        return self.derivatives['dll_dTheta']

    def predict(self, p):
        self.forward_propagate(p)  # incorporate compute_weights()
        self.compute_model()
        return self.parameters['model']

    @st_time
    def fit(self, P, data):
        # for epoch in range(epochs):
        ll = 0  # stores the ll
        n_c = 0  # stores the number of correct predictions
        failed = 0
        for i in range(0, P.shape[0]):
            childLogger.info("*********************************\n"
                             "Initial value of parameter's set P[{}]={}".format(i, P[i]))
            data_predicted_first = self.predict(P[i])
            self.compute_ll(data)
            self.compute_derivatives_dphi(data)
            self.compute_derivatives_dTheta()
            childLogger.info("Initial Likelihood={}\nInitial predicted data: {}\n".format(self.parameters['ll'],
                                                                                          data_predicted_first))
            self.update_parameters(0.1, 500)
            popt = list(self.parameters['Theta'].values())
            check_bounds = [0 if value > upper or value < lower else 1 for (upper, value, lower)
                            in zip(self.upper_bound, popt,
                                   self.lower_bound)]
            if np.isnan(popt).any() or not all(check_bounds):
                childLogger.info("Optimized params beyond bounds, continue")
                failed += 1
                continue

            data_predicted_opt = self.predict(popt)
            self.compute_ll(data)
            ll = self.parameters['ll']
            theta = dadi.Inference.optimal_sfs_scaling(data_predicted_opt, data)
            childLogger.info("data:{}\ndata_predicted:{}\nLikelihood:{}"
                             "\nValue of scaling theta={}".format(data, data_predicted_opt, ll, theta))
            if np.isclose(data_predicted_opt * theta, data, atol=5.0).all():
                n_c += 1
                childLogger.debug("isclose, n_c={}, with parameters:{}\nLikelihood:{}".
                                  format(n_c, popt, ll))
                # childLogger.info("Grad", list(self.derivatives['dll_dTheta']))
        return popt


"""
# Importing the dataset
dataset = dadi.Spectrum.from_file('fs_data.fs')
ns = dataset.sample_sizes  # mask corners
# 'nu': population size, 'gamma': selection coefficient, 'h': dominance coefficient, 'beta': breeding ratio, 'theta0': 1
upper_bound = [100, 1, 2, 10, 1]
lower_bound = [1e-2, 1e-2, -1, 1e-2, 1]
number_of_traininigs = 5000
# (training) set of vectors of parameters P:
# this is our initial guesses for the parameters, which is somewhat arbitrary
#
P = list()
for low, high in zip(lower_bound, upper_bound):
    # column = scipy.stats.norm(low=low, high=high, size=number_of_traininigs)
    column = np.random.uniform(low=low, high=high, size=number_of_traininigs)
    P.append(column)

P = np.asarray(P).T
# # popt = [2, 0.5, 0.5, 1, 1]
# {
#     'nu': 27.57729022447911, 'gamma': 0.4399618218742022, 'h': 0.7428075146906175, 'beta': 1.4079518902437447,
#     'theta0': 1.0
#     }
# P = np.asarray([[27.5, 0.4, 0.7, 1.4, 1]])
# Defining the model architecture
time_architecture_initial = 0
time_architecture_last = 3
pts = 30
xx = dadi.Numerics.default_grid(pts)
# Creating the Network
adjointer = NeuralNetwork(time_architecture_initial, time_architecture_last, ns, pts, xx, upper_bound,
                          lower_bound)
print(P)
print(P.shape[0])
print(P[0])
childLogger = logging.getLogger(__name__)
childLogger.addHandler(logging.FileHandler('neural_backp.log'))
# logger = logging.getLogger()
childLogger.setLevel(10)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# childLogger.addFormater
childLogger.debug("number_of_traininigs={}, upper bound={}, lower_bound={}, time_initial={},\ntime_last={}, pts={}"
                  "lr={}, iter={}".format(number_of_traininigs, upper_bound, lower_bound, time_architecture_initial,
                                          time_architecture_last, pts, "1e7", 3500))

adjointer.fit(P, dataset, 1e7, 3500)
"""
