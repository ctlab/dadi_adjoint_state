import dadi
import numpy as np
import scipy
from scipy.integrate import trapz
import scipy.special, scipy.sparse


def _Vfunc(x, nu, beta=1):
    return 1. / nu * x * (1 - x) * (beta + 1.) ** 2 / (4. * beta)


def _Mfunc1D(x, gamma, h):
    return gamma * 2 * x * (h + (1 - 2 * h) * x) * (1 - x)


def _Vfunc_dnu(x, nu, beta=1):
    return (- x * (1 - x) * (beta + 1.) ** 2) / (nu ** 2 * 4. * beta)


def _Vfunc_dbeta(x, nu, beta=1):
    return (2 * x * (1 - x) * (beta + 1.) * 4 * nu * beta - 4 * nu * x * (1 - x) * (beta + 1) ** 2) / (
            16 * nu ** 2 * beta ** 2)


def _Mfunc1D_dgamma(x, h):
    return 2 * x * (h + (1 - 2 * h) * x) * (1 - x)


def _Mfunc1D_dh(x, gamma):
    return 2 * gamma * x * (2 * x ** 2 - 3 * x + 1)


def calc_tridiag_matrix(phi, dfactor, MInt, M, V, dx, nu, delj=0.5):
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
    return tridiag_matrix


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


def calc_dtridiag_dgamma(phi_initial, dfactor, dM_dgamma_Int, M, dx, delj=0.5):
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


def calc_dtridiag_dh(phi_initial, dfactor, dM_dh, M, dx, delj=0.5):
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
    This derivative play role of activation function
    (dA)^-1/dTheta = -A^(-1)(dA/dTheta)A^(-1)
    """
    result = np.matmul(-inverse_tridiag, dtridiag_dTheta)
    result = np.matmul(result, inverse_tridiag)
    return result


def calc_injected_and_next_phi(previous_phi, tridiag_matrix, this_dt, xx, theta0):
    injected_phi = dadi.Integration._inject_mutations_1D(previous_phi, this_dt, xx, theta0)
    next_phi = np.matmul(np.linalg.inv(tridiag_matrix), injected_phi)  # scipy.linalg.inv(tridiag_matrix)
    return injected_phi, next_phi


def _from_phi_1D_direct_dphi_directly(n, xx, mask_corners=True,
                                      het_ascertained=None):
    """
    Compute derivative from sample Spectrum from population frequency distribution phi.
    See from_phi for explanation of arguments.
    """
    n = round(n)
    data = np.zeros(n + 1)
    for ii in range(0, n + 1):
        factorx = scipy.special.comb(n, ii) * xx ** ii * (1 - xx) ** (n - ii)
        if het_ascertained == 'xx':
            factorx *= xx * (1 - xx)
        data[ii] = trapz(factorx, xx)
    return dadi.Spectrum(data, mask_corners=mask_corners)


def dll_dphi(model, data, ns, xx):
    """ analytical derivative"""
    dmodel_dphi = _from_phi_1D_direct_dphi_directly(ns[0], xx)
    return dmodel_dphi * (data / model - 1)


class NeuralNetwork(object):
    def __init__(self, timeline_architecture_initial, timeline_architecture_last, ns, pts):
        # architecture - numpy array with ith element representing the number of neurons in the ith layer.
        """
        ns: Sample size of resulting Spectrum
        pts: Number of grid points to use in integration.
        ns = (n1,n2): Size of fs to generate.
        pts: Number of points to use in grid for evaluation.
        there should be
        phi.ndim == len(ns) == len(xxs), where
        phi: P-dimensional population frequency distribution.
        ns: Sequence of P sample sizes for each population.
        xxs: Sequence of P one-dimesional grids on which phi is defined.
        """
        self.T = timeline_architecture_last  # L corresponds to the last layer of the network
        self.initial_t = timeline_architecture_initial
        self.pts = pts  # n (pts) stores the number of neurons in each layer
        self.ns = ns
        # input_size is the number of neurons in the first layer i.e. pts[0]
        # output_size is the number of neurons in the last layer i.e. pts[L]
        # Parameters will store the network parameters,
        # nu - (population size) specifies the relative size of this ancestral population to the
        # reference population. Most often, the reference population is the ancestral, so nu defaults to 1.
        # by default the mutation parameter θ=4Nrefμ is set to 1. If you wish to set a fixed value of θ=4N0μ
        # in your analysis, that information must be provided to the initial ϕ creation function and the
        # Integration functions. When fixing θ, every Integration function must be told what the reference θ is,
        # using the option theta0
        # gamma - selection coefficient
        # h - dominance coefficient
        # ns - list of sample sizes
        # i.e. the weights (tridiagonal matrix A) and biases (phi_injected)
        # time in dadi models is in genetic units: times are given in units of 2Nref generations - also it is the
        # number of layers
        self.xx = dadi.Numerics.default_grid(self.pts)
        self.dx = np.diff(self.xx)
        self.dfactor = dadi.Integration._compute_dfactor(self.dx)
        self.delj = dadi.Integration._compute_delj(self.dx, self.parameters['MInt'], self.parameters['VInt'])

        self.parameters = {'theta': {'nu': 2, 'gamma': 0.5, 'h': 0.5, 'beta': 1, 'theta0': 1}}
        self.parameters['phi0'] = dadi.PhiManip.phi_1D(self.xx, self.parameters['theta']['nu'],
                                                       self.parameters['theta']['theta0'],
                                                       self.parameters['theta']['gamma'], self.parameters['theta']['h'],
                                                       self.parameters['theta']['beta'])
        # just for initialization
        self.parameters['phi_injected'] = dict()  # self.parameters['phi0']
        self.parameters['phi'] = dict()  # self.parameters['phi0']

        # Initialize the likelihood:
        self.parameters['ll'] = np.zeros(self.ns[0] + 1)
        self.parameters['model'] = np.zeros(self.ns[0] + 1)

        self.derivatives = {'dF_dphi': -1 * np.ones(self.ns[0])}  # self.pts seems to be more righter

    def compute_weights(self):
        self.parameters['M'] = _Mfunc1D(self.xx, self.parameters['theta']['gamma'], self.parameters['theta']['h'])
        self.parameters['MInt'] = _Mfunc1D((self.xx[:-1] + self.xx[1:]) / 2, self.parameters['theta']['gamma'],
                                           self.parameters['theta']['h'])
        self.parameters['dM_dgamma'] = _Mfunc1D_dgamma((self.xx[:-1] + self.xx[1:]) / 2, self.parameters['theta']['h'])  # Int
        self.parameters['dM_dh'] = _Mfunc1D_dh((self.xx[:-1] + self.xx[1:]) / 2, self.parameters['theta']['gamma'])  # Int

        self.parameters['V'] = _Vfunc(self.xx, self.parameters['theta']['nu'], self.parameters['theta']['beta'])
        self.parameters['VInt'] = _Vfunc((self.xx[:-1] + self.xx[1:]) / 2, self.parameters['theta']['nu'],
                                         self.parameters['theta']['beta'])
        self.parameters['dV_dnu'] = _Vfunc_dnu(self.parameters['xx'], self.parameters['theta']['nu'],
                                               self.parameters['theta']['beta'])
        self.parameters['dV_dbeta'] = _Vfunc_dbeta(self.parameters['xx'], self.parameters['theta']['nu'],
                                                   self.parameters['theta']['beta'])

        # Initialize the tridiagonal matrix:
        self.parameters['A'] = calc_tridiag_matrix(self.parameters['phi0'], self.dfactor, self.parameters['MInt'],
                                                   self.parameters['M'], self.parameters['V'], self.dx, self.delj)
        self.parameters['A_inv'] = calc_inverse_tridiag_matrix(self.parameters['A'])
        self.parameters['dA_dnu'] = calc_dtridiag_dnu(self.parameters['phi0'], self.dfactor, self.parameters['dV_dnu'],
                                                      self.dx, self.parameters['theta']['nu'], self.parameters['M'])
        self.parameters['dA_dbeta'] = calc_dtridiag_dbeta(self.parameters['phi0'], self.dfactor,
                                                          self.parameters['dV_dbeta'], self.dx)
        self.parameters['dA_dgamma'] = calc_dtridiag_dgamma(self.parameters['phi0'], self.dfactor,
                                                            self.parameters['dM_dgamma'], self.parameters['M'],
                                                            self.dx, self.delj)
        self.parameters['dA_dh'] = calc_dtridiag_dh(self.parameters['phi0'], self.dfactor,
                                                    self.parameters['dM_dh'], self.parameters['M'], self.dx,
                                                    self.delj)
        self.parameters['dA_dtheta'] = get_dtridiag_dTheta(self.parameters['dA_dnu'], self.parameters['dA_dbeta'],
                                                           self.parameters['dA_dgamma'], self.parameters['dA_dh'])
        self.parameters['dA_inv_dtheta'] = get_dtridiag_inverse_dTheta(self.parameters['A_inv'],
                                                                       self.parameters['dA_dtheta'])
        """
        self.derivatives = {'dF_dtheta': np.zeros([self.parameters['dA_inv_dtheta'].shape[0], self.ns[0]],
                                                  dtype=float),
                            'dll_dtheta': ...,
                            'dF_dphi': -1 * np.ones(ns),
                            'dll_dphi': np.zeros(self.ns[0] + 1),
                            'adjoint_field': np.zeros(self.ns[0])}
        """

    def forward_propagate(self):
        # Calculate the activations (phi_injected) and output (phi) for every layer t
        dt = dadi.Integration._compute_dt(self.dx, self.parameters['theta']['nu'], [0], self.parameters['theta']['gamma'],
                                          self.parameters['theta']['h'])
        current_t = self.initial_t
        previous_phi = self.parameters['phi0']
        while current_t < self.T:
            this_dt = min(dt, self.T - current_t)
            self.parameters['phi_injected']['phi_injected_' + str(this_dt)], self.parameters['phi']['phi_' + str(
                this_dt)] = \
                calc_injected_and_next_phi(previous_phi, self.parameters['A'], this_dt, self.xx,
                                           self.parameters['theta']['theta0'])
            # self.parameters['phi_injected'], self.parameters['phi'] = \
            #    calc_injected_and_next_phi(previous_phi, self.parameters['A'], this_dt, self.xx, self.parameters[
            #    'theta0'])
            previous_phi = self.parameters['phi']['phi_' + str(this_dt)]
            current_t += this_dt

    def compute_model(self):
        self.parameters['model'] = dadi.Spectrum.from_phi(list(self.parameters['phi'].values())[-1], self.ns,
                                                          self.parameters['xx'], force_direct=True)

    def compute_ll(self, data):
        # for access the last value from dict use list(self.parameters['phi'].values())[-1]
        self.parameters['ll'] = data * np.log(self.parameters['model']) - self.parameters['model'] - np.log(data)

    def compute_derivatives(self, data):
        """
        # Partial derivatives of the cost function with respect to phi[L], theta + adjoint field:
        """
        self.derivatives['dll_dphi'] = dll_dphi(self.parameters['model'], data, self.ns, self.xx)
        # adjoint-state lagrange multipliers - adjoint_field
        self.derivatives['adjoint_field'] = np.full(self.ns[0], np.matmul(self.derivatives['dF_dphi'].T,
                                                                          np.asarray(self.derivatives['dll_dphi'])[1:]))
        shorted_phi_inj = list(self.parameters['phi_injected'].items())[:-1]
        shorted_phi_inj = dict(shorted_phi_inj)  # [::-1] for reverse mode
        for phi_inj_name, phi_inj_value in shorted_phi_inj.items():
            self.derivatives['dF_dtheta'] += np.matmul(self.parameters['dA_inv_dtheta'], phi_inj_value)
            self.derivatives['dF_dtheta'] = np.matmul(self.derivatives['dF_dtheta'],
                                                      self.parameters['A_inv'])
        self.derivatives['dll_dtheta'] = np.matmul(self.derivatives['dF_dtheta'], self.derivatives['adjoint_field'])
        # i = phi_inj_name.split('_')[-1]
        # self.derivatives['dF_dtheta']['dF_dtheta' + i] += np.matmul(self.parameters['dA_inv_dtheta'],
        # phi_inj_value)
        # self.derivatives['dF_dtheta']['dF_dtheta' + i] = np.matmul(self.derivatives['dF_dtheta']['dF_dtheta' + i],
        #                                                           self.parameters['A_inv'])

    def update_parameters(self, lr):
        dt = dadi.Integration._compute_dt(self.dx, self.parameters['theta']['nu'], [0],
                                          self.parameters['theta']['gamma'],
                                          self.parameters['theta']['h'])
        current_t = self.initial_t
        while current_t < self.T:
            this_dt = min(dt, self.T - current_t)
            for param in self.parameters['theta'].keys():
                self.parameters['theta'][param] += lr * self.derivatives['dll_dtheta']
            current_t += this_dt
        self.compute_weights()

    def predict(self):
        self.forward_propagate()
        self.compute_model()
        return self.parameters['model']

    def fit(self, P, data, num_iter, lr=0.01):
        for iter in range(0, num_iter):
            ll = 0  # stores the ll
            n_c = 0  # stores the number of correct predictions
            for i in range(0, P.shape[0]):
                p = P[i].reshape((P[i].size, 1))

                self.forward_propagate(p)
                self.compute_ll(data)
                self.compute_derivatives(data)
                self.update_parameters(lr)

                ll += self.parameters['ll']
                data_predicted = self.predict()


# Importing the dataset
dataset = dadi.Spectrum.from_file('fs_data.fs')
ns = dataset.sample_sizes  # mask corners
# 'nu': population size, 'gamma': selection coefficient, 'h': dominance coefficient, 'beta': breeding ratio, 'theta0': 1
upper_bound = [100, 1, 1, 10, 1]
lower_bound = [1e-2, 1e-2, 1e-2, 1e-2, 1]
number_of_traininigs = 10
# set of vectors of parameters P
P = list()
for low, high in zip(lower_bound, upper_bound):
        column = np.random.uniform(low=low, high=high, size=number_of_traininigs)
        P.append(column)

P = np.asarray(P).T

# This is our initial guess for the parameters, which is somewhat arbitrary.
# p0 = [2, 0.5, 0.5, 1, 1]
# Make the extrapolating version of our demographic model function.
# func_ex = dadi.Numerics.make_extrap_log_func(func)

# Perturb our parameters before optimization. This does so by taking each
# parameter a up to a factor of two up or down.
# p0 = dadi.Misc.perturb_params(p0, fold=1, upper_bound=upper_bound,
#                              lower_bound=lower_bound)

# Defining the model architecture
timeline_architecture_initial = 0
timeline_architecture_last = 70
pts = 19
# Creating the Network
adjointer = NeuralNetwork(timeline_architecture_initial, timeline_architecture_last, ns, pts)
# Training the classifier
adjointer.fit(P, dataset, 100)
# Predicting:
n_c = 0
