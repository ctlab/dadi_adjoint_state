import numpy as np
import math
import scipy
from scipy import sparse
from scipy.integrate import trapz
from numpy import newaxis as nuax
import logging
import scipy.optimize
import matplotlib.pyplot as plt
import dadi
import os
from scipy.optimize import BFGS

#: Controls timestep for integrations. This is a reasonable default for
#: gridsizes of ~60. See set_timescale_factor for better control.


timescale_factor = 1e-3

#: Whether to use old timestep method, which is old_timescale_factor * dx[0].
use_old_timestep = False
#: Factor for told timestep method.
old_timescale_factor = 0.1

#: Controls use of Chang and Cooper's delj trick, which seems to lower accuracy.
use_delj_trick = True


def simple_grid(pts):
    """
    pts: Number of grid points to use in integration.
    """
    L = 1.0  # length of the 1D domain
    # Define the grid point coordinates.
    grid = np.linspace(0.0, L, num=pts)
    return grid
#
# Here are the python versions of the population genetic functions and its derivatives.
#
def _Vfunc(x, nu, beta=1):
    return 1. / nu * x * (1 - x) * (beta + 1.) ** 2 / (4. * beta)

def _Mfunc1D(x, gamma, h):
    return gamma * 2 * x * (h + (1 - 2 * h) * x) * (1 - x)

def _Vfunc_dnu(x, nu, beta=1):
    return (- x * (1 - x) * (beta + 1.) ** 2) / (nu**2 * 4. * beta)

def _Vfunc_dbeta(x, nu, beta=1):
    return (2*x * (1 - x) * (beta + 1.)*4*nu*beta - 4*nu*x*(1 - x)*(beta + 1)**2)/(16*nu**2*beta**2)

def _Mfunc1D_dgamma(x, h):
    return 2 * x *(h + (1 - 2 * h) * x) * (1 - x)

def _Mfunc1D_dh(x, gamma):
    return 2 * gamma * x * (2*x**2 - 3*x + 1)

def _compute_delj(dx, MInt, VInt, axis=0):
    r"""
    Chang an Cooper's \delta_j term. Typically we set this to 0.5.
    """
    # Chang and Cooper's fancy delta j trick...
    if use_delj_trick:
        # upslice will raise the dimensionality of dx and VInt to be appropriate
        # for functioning with MInt.
        upslice = [nuax for ii in range(MInt.ndim)]
        upslice [axis] = slice(None)

        #wj = 2 *MInt*dx[upslice]
        wj = MInt * dx[upslice]
        epsj = np.exp(wj/VInt[upslice])
        delj = 1/wj - 1/(epsj - 1) #(-epsj*wj + epsj * VInt[upslice] - VInt[upslice])/(wj - epsj*wj)
        # These where statements filter out edge case for delj
        delj = np.where(np.isnan(delj), 0.5, delj)
        delj = np.where(np.isinf(delj), 0.5, delj)
    else:
        delj = 0.5
    return delj

def _compute_dfactor(dx):
    r"""
    \Delta_j from the paper.
    """
    # Controls how we take the derivative of the flux. The values here depend
    #  on the fact that we're defining our probability integral using the
    #  trapezoid rule.
    dfactor = np.zeros(len(dx) + 1)
    dfactor[1:-1] = 2 / (dx[:-1] + dx[1:])
    dfactor[0] = 2 / dx[0]
    dfactor[-1] = 2 / dx[-1]
    return dfactor


def _compute_delj(dx, MInt, VInt, axis=0):
    r"""
    Chang an Cooper's \delta_j term. Typically we set this to 0.5.
    """
    # Chang and Cooper's fancy delta j trick...
    if use_delj_trick:
        # upslice will raise the dimensionality of dx and VInt to be appropriate
        # for functioning with MInt.
        upslice = [nuax for ii in range(MInt.ndim)]
        upslice[axis] = slice(None)
        wj = 2 * MInt * dx[tuple(upslice)]
        #wj = MInt * dx[upslice]
        epsj = np.exp(wj / VInt[tuple(upslice)])
        delj = (-epsj * wj + epsj * VInt[tuple(upslice)] - VInt[tuple(upslice)]) / (wj - epsj * wj)
        # These where statements filter out edge case for delj
        delj = np.where(np.isnan(delj), 0.5, delj)
        delj = np.where(np.isinf(delj), 0.5, delj)
    else:
        delj = 0.5
    print("delj", delj)
    return delj


def _inject_mutations_1D(phi, dt, xx, theta0):
    # Inject novel mutations for a timestep.

    phi[1] += dt / xx[1] * theta0 / 2 * 2 / (xx[2] - xx[0])
    return phi


def _compute_dt(dx, nu, ms, gamma, h):
    # Compute the appropriate timestep given the current demographic params.

    # This is based on the maximum V or M expected in this direction. The
    # timestep is scaled such that if the params are rescaled correctly by a
    # constant, the exact same integration happens. (This is equivalent to
    # multiplying the eqn through by some other 2N...)

    if use_old_timestep:
        return old_timescale_factor * dx[0]

    # These are the maxima for V_func and M_func over the domain
    # For h != 0.5, the maximum of M_func is not easy analytically. It is close
    # to the 0.5 or 0.25 value, though, so we use those as an approximation.

    # It might seem natural to scale dt based on dx[0]. However, testing has
    # shown that extrapolation is much more reliable when the same timesteps
    # are used in evaluations at different grid sizes.
    maxVM = max(0.25 / nu, sum(ms), \
                abs(gamma) * 2 * max(np.abs(h + (1 - 2 * h) * 0.5) * 0.5 * (1 - 0.5),
                                     np.abs(h + (1 - 2 * h) * 0.25) * 0.25 * (1 - 0.25)))
    if maxVM > 0:
        dt = timescale_factor / maxVM
    else:
        dt = np.inf
    if dt == 0:
        raise ValueError('Timestep is zero. Values passed in are nu=%f, ms=%s,'
                         'gamma=%f, h=%f.' % (nu, str(ms), gamma, h))
    return dt


def calc_injected_and_next_phi(previous_phi, trans_matrix, this_dt, xx, theta0):
    injected_phi = _inject_mutations_1D(previous_phi, this_dt, xx, theta0)
    next_phi = np.matmul(scipy.linalg.inv(trans_matrix), injected_phi)
    return injected_phi, next_phi


def calc_tridiag_matrix(dfactor, MInt, M, V, dx, delj=0.5):
    a = np.zeros(initial_phi.shape)
    a[1:] += dfactor[1:] * (-MInt * delj - V[:-1] / (2 * dx))
    c = np.zeros(initial_phi.shape)
    c[:-1] += -dfactor[:-1] * (-MInt * (1 - delj) + V[1:] / (2 * dx))
    b = np.zeros(initial_phi.shape)
    b[:-1] += -dfactor[:-1] * (-MInt * delj - V[:-1] / (2 * dx))
    b[1:] += dfactor[1:] * (-MInt * (1 - delj) + V[1:] / (2 * dx))

    if(M[0] <= 0):
        b[0] += (0.5/nu - M[0])*2/dx[0]
    if(M[-1] >= 0):
        b[-1] += -(-0.5/nu - M[-1])*2/dx[-1]

    tridiag_matrix = scipy.sparse.diags([-a[1:], b, -c[:-1]], [-1, 0, 1]).toarray()

    return tridiag_matrix

def calc_inverse_tridiag_matrix(matrix):
    return np.linalg.inv(matrix)

def calc_dtridiag_dnu(dfactor, dVdnu, dx, M):
    a = np.zeros(initial_phi.shape)
    a[1:] += -dfactor[1:] * dVdnu[:-1] / (2 * dx)
    c = np.zeros(initial_phi.shape)
    c[:-1] += -dfactor[:-1] * dVdnu[1:] / (2 * dx)
    b = np.zeros(initial_phi.shape)
    b[:-1] += dfactor[:-1] * dVdnu[:-1] / (2 * dx)
    b[1:] += dfactor[1:] * dVdnu[1:] / (2 * dx)

    if(M[0] <= 0):
        b[0] += 1/(dx[0]*nu**2)
    if(M[-1] >= 0):
        b[-1] += -1/(dx[-1]*nu**2)

    dtridiag_dnu = scipy.sparse.diags([-a[1:], b, -c[:-1]], [-1, 0, 1]).toarray()

    return dtridiag_dnu

def calc_dtridiag_dbeta(dfactor, dVdbeta, dx):
    a = np.zeros(initial_phi.shape)
    a[1:] += -dfactor[1:] * dVdbeta[:-1] / (2 * dx)
    c = np.zeros(initial_phi.shape)
    c[:-1] += -dfactor[:-1] * dVdbeta[1:] / (2 * dx)
    b = np.zeros(initial_phi.shape)
    b[:-1] += dfactor[:-1] * dVdbeta[:-1] / (2 * dx)
    b[1:] += dfactor[1:] * dVdbeta[1:] / (2 * dx)
    #derivarive from additional part of b by beta is zero

    dtridiag_dbeta = scipy.sparse.diags([-a[1:], b, -c[:-1]], [-1, 0, 1]).toarray()

    return dtridiag_dbeta

def calc_dtridiag_dgamma(dfactor, dMdgamma_Int, M, dx, delj=0.5):
    a = np.zeros(initial_phi.shape)
    a[1:] += -dfactor[1:] * delj * dMdgamma_Int
    c = np.zeros(initial_phi.shape)
    c[:-1] += dfactor[:-1] * (1 - delj) * dMdgamma_Int
    b = np.zeros(initial_phi.shape)
    b[:-1] += dfactor[:-1] * delj * dMdgamma_Int
    b[1:] += -dfactor[1:] * (1 - delj) * dMdgamma_Int

    if(M[0] <= 0):
        b[0] += -dMdgamma_Int[0] * 2 / dx[0]
    if(M[-1] >= 0):
        b[-1] += dMdgamma_Int[-1] * 2 / dx[-1]

    dtridiag_dgamma = scipy.sparse.diags([-a[1:], b, -c[:-1]], [-1, 0, 1]).toarray()

    return dtridiag_dgamma

def calc_dtridiag_dh(dfactor, dMdh, M, dx, delj=0.5):
    a = np.zeros(initial_phi.shape)
    a[1:] += -dfactor[1:] * delj * dMdh
    c = np.zeros(initial_phi.shape)
    c[:-1] += dfactor[:-1] * (1 - delj) * dMdh
    b = np.zeros(initial_phi.shape)
    b[:-1] += dfactor[:-1] * delj * dMdh
    b[1:] += -dfactor[1:] * (1 - delj) * dMdh

    if(M[0] <= 0):
        b[0] += -dMdh[0]*2/dx[0]
    if(M[-1] >= 0):
        b[-1] += dMdh[-1]*2/dx[-1]

    dtridiag_dh = scipy.sparse.diags([-a[1:], b, -c[:-1]], [-1, 0, 1]).toarray()

    return dtridiag_dh

def get_dtridiagdTheta(dtridiag_dnu, dtridiag_dbeta, dtridiag_dgamma, dtridiag_dh):
    return np.array([dtridiag_dnu, dtridiag_dbeta, dtridiag_dgamma, dtridiag_dh])

def get_dtridiag_inverse_dTheta(inverse_tridiag, dtridiagdTheta):
    result = np.matmul(-inverse_tridiag, dtridiagdTheta)
    result = np.matmul(result, inverse_tridiag)
    return result


def feedforward(theta, previous_phi, tridiag_matrix, inverse_tridiag, xx, ns, dx, dfactor, T=3, theta0=1, initial_t=0,
                delj=0.5):
    nu, gamma, h, beta = theta
    M = _Mfunc1D(xx, gamma, h)
    #MInt = _Mfunc1D((xx[:-1] + xx[1:]) / 2, gamma, h)
    dMdgamma_Int = _Mfunc1D_dgamma((xx[:-1] + xx[1:]) / 2, h)
    dMdh_Int = _Mfunc1D_dh((xx[:-1] + xx[1:]) / 2, gamma)
    dVdnu = _Vfunc_dnu(xx, nu, beta)
    dVdbeta = _Vfunc_dbeta(xx, nu, beta)
    dtridiag_dnu = calc_dtridiag_dnu(dfactor, dVdnu, dx, M)
    dtridiag_dbeta = calc_dtridiag_dbeta(dfactor, dVdbeta, dx)
    dtridiag_dgamma = calc_dtridiag_dgamma(dfactor, dMdgamma_Int, M, dx, delj)
    dtridiag_dh = calc_dtridiag_dh(dfactor, dMdh_Int, M, dx, delj)
    dtridiagdTheta = get_dtridiagdTheta(dtridiag_dnu, dtridiag_dbeta, dtridiag_dgamma, dtridiag_dh)
    dtridiag_inverse_dTheta = get_dtridiag_inverse_dTheta(inverse_tridiag, dtridiagdTheta)
    dFdtheta = np.zeros((dtridiag_inverse_dTheta.shape[0], ns[0]))

    dt = _compute_dt(dx, nu, [0], gamma, h)
    current_t = initial_t
    while current_t < T:
        this_dt = min(dt, T - current_t)
        phi_injected, phi = calc_injected_and_next_phi(previous_phi, tridiag_matrix, this_dt, xx, theta0)
        previous_phi = phi
        current_t += this_dt
        dFdtheta += np.matmul(dtridiag_inverse_dTheta, phi_injected)
        dFdtheta = np.matmul(dFdtheta, inverse_tridiag)
    #inject
    dFdtheta = np.matmul(dFdtheta, tridiag_matrix)

    if not phi.any():
        logging.error('Phi not counted')
        return

    return phi, dFdtheta


def calc_target_grad(dFdtheta, adjoint_field):
    target_grad = np.matmul(dFdtheta, adjoint_field)
    #logging.debug("target_grad: {0}".format(target_grad))
    norm = np.linalg.norm(target_grad)
    #logging.debug("target_grad norm: {0}".format(norm))
    return target_grad


def calc_model_AFS(phi, xx, ns):
    """
        Compute sample spectrum from population frequency distribution phi.
        ns: Sequence of P sample sizes for each population.
        xx: Sequence of P one-dimensional grids on which phi is defined.
    """
    model_AFS = np.zeros(ns)
    for ii in range(0, ns):
        factorx = scipy.special.comb(ns, ii) * xx ** ii * (1 - xx) ** (ns - ii)
        model_AFS[ii] = trapz(factorx * phi)
    return model_AFS


def calc_dmodel_AFS_dphi(xx, ns):
    """  """
    deriv_model_AFS = np.zeros(ns)
    for ii in range(0, ns):
        factorx = scipy.special.comb(ns, ii) * xx ** ii * (1 - xx) ** (ns - ii)
        deriv_model_AFS[ii] = trapz(factorx)
    return deriv_model_AFS

def calc_objective_func(phi, xx, ns, observed_spectrum):
    model = calc_model_AFS(phi, xx, ns)
    obj_func = observed_spectrum[1:] * np.log(model) - model - np.log(observed_spectrum[1:])
    return obj_func

def calc_objective_func_from_theta(theta, initial_phi, tridiag, inverse_tridiag, xx, ns, dx, dfactor, observed_spectrum,
                                       T, theta0, initial_t, delj):
    phi, _ = feedforward(theta, initial_phi, tridiag, inverse_tridiag, xx, ns, dx, dfactor, T=3, theta0=1, initial_t=0, delj=delj)
    model = calc_model_AFS(phi, xx, ns[0])

    obj_func = observed_spectrum[1:] * np.log(model) - model - np.log(observed_spectrum[1:])
    return obj_func


def functional_F(current_phi, next_phi):
    return next_phi - current_phi


def dfunctionalF_dphi(ns):
    """ ns - size of grid"""
    return -1 * np.ones(ns)


def ascent(theta, dFdtheta, adjoint_field, phi, tridiag, inverse_tridiag, xx, ns, dx,dfactor, T=3, theta0=1, initial_t=0):
    learning_rate = 0.1
    target_grad = calc_target_grad(dFdtheta, adjoint_field)
    for i in range(10):
    #while np.linalg.norm(target_grad) >= 0.5:
        theta_optimized = theta + learning_rate * target_grad
        theta = theta_optimized
        phi, dFdtheta = feedforward(theta, phi, tridiag, inverse_tridiag, xx, ns, dx, dfactor, T=20, theta0=1,
                                    initial_t=0)
        target_grad = calc_target_grad(dFdtheta, adjoint_field)
        #logging.debug("theta: {0}".format(theta))
    return theta

def plot_function(func_output):
    plt.figure(figsize=(10,7))
    x = np.linspace(-1, 2, len(func_output))
    y = func_output
    plt.plot(x, y)
    plt.ylabel("Y")
    plt.xlabel("X")
    #plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    plt.scatter(x, y, linewidths=5)
    #plt.scatter([3/8], [F(3/8)], lw=5)
    plt.show()


def main(observed_spectrum, ns, xx, nu, gamma, h, beta, initial_phi):
    logger = logging.getLogger()
    logger.setLevel(10)
    #logging.basicConfig(format='[%(asctime)s] %(levelname).1s %(message)s', level=logging.DEBUG)

    logger.debug("observed spectrum: {0}\n, xx: {1}\n, nu: {2}\n, gamma: {3}\n,"
                  " h: {4}\n, beta: {5}\n, initial_phi: {6}\n".format(observed_spectrum, xx, nu, gamma, h, beta, initial_phi))


    theta = np.array([nu, gamma, h, beta])

    M = _Mfunc1D(xx, gamma, h)
    MInt = _Mfunc1D((xx[:-1] + xx[1:]) / 2, gamma, h)
    dMdgamma = _Mfunc1D_dgamma((xx[:-1] + xx[1:]) / 2, h)  # Int
    dMdh = _Mfunc1D_dh((xx[:-1] + xx[1:]) / 2, gamma)  # Int
    logging.debug("M: {0},\n MInt: {1},\n dMdgamma: {2},\n dMdh: {3}".format(M, MInt, dMdgamma, dMdh))

    V = _Vfunc(xx, nu, beta)
    VInt = _Vfunc((xx[:-1] + xx[1:]) / 2, nu, beta=beta)
    dVdnu = _Vfunc_dnu(xx, nu, beta)
    dVdbeta = _Vfunc_dbeta(xx, nu, beta)
    logging.debug("V: {0} \n, dVdnu: {1} \n, dVdbeta: {2} \n".format(V, dVdnu, dVdbeta))
    dx = np.diff(xx)
    dfactor = _compute_dfactor(dx)
    delj = _compute_delj(dx, MInt, VInt)
    logging.debug("delj: {0}".format(delj))
    tridiag = calc_tridiag_matrix(dfactor, MInt, M, V, dx, delj)
    logging.debug("dfactor: {0}\n, tridiag: \n {1}, determinant: {2}".format(dfactor, tridiag, np.linalg.det(tridiag)))
    inverse_tridiag = calc_inverse_tridiag_matrix(tridiag)
    logging.debug("inverse_tridiag: \n {0}, shape {1} \n".format(inverse_tridiag, inverse_tridiag.shape))

    phi, dFdtheta = feedforward(theta, initial_phi, tridiag, inverse_tridiag, xx, ns, dx, dfactor, T=20, theta0=1,
                                initial_t=0, delj=delj)
    logging.debug("phi from feedforward: {0}\n, dFdtheta (third derivative for ASM): \n {1}".format(phi, dFdtheta))

    model_AFS = calc_model_AFS(phi, xx, ns[0])
    logging.debug("model AFS - spectrum from phi: {0}\n".format(model_AFS))
    derivative_model_AFS = calc_dmodel_AFS_dphi(xx, ns[0])
    logging.debug("derivative from model AFS - (first derivative for ASM): {0}\n".format(derivative_model_AFS))
    dFdphi = dfunctionalF_dphi(ns[0])
    logging.debug("dFdphi - (second derivative for ASM): {0}\n".format(dFdphi))


    adjoint_field = np.full(ns, np.matmul(dFdphi.T, derivative_model_AFS))
    logging.debug("adjoint-state lagrange multipliers: {0}".format(adjoint_field))
    target_grad = calc_target_grad(dFdtheta, adjoint_field)
    logging.debug("target_grad: {0}\n".format(target_grad))

    objective_functional = calc_objective_func(phi, xx, ns[0], observed_spectrum)
    logging.debug("objective_functional: {0}\n".format(objective_functional))
    objective_functional_from_theta = calc_objective_func_from_theta(theta, initial_phi, tridiag, inverse_tridiag, xx,
                                                                     ns, dx, dfactor, observed_spectrum, T=20, theta0=1,
                                                                    initial_t=0, delj=delj)
    logging.debug("objective_functional_from_theta: {0}\n".format(objective_functional_from_theta))


    theta_opt = ascent(theta, dFdtheta, adjoint_field, phi, tridiag, inverse_tridiag, xx, ns, dx, dfactor)
    logging.debug("theta_opt: {0}".format(theta_opt))
    print(len(observed_spectrum))
    opt_res = scipy.optimize.minimize(calc_objective_func_from_theta, theta,
                                                              (initial_phi, tridiag, inverse_tridiag, xx, ns, dx, dfactor,
                                                                observed_spectrum, 20, 1, 0, delj))
    logging.debug("opt_res item: {0}".format(opt_res.item()))
    logging.debug("opt_res: {0}".format(opt_res))


    #plot_function(V)
    #plot_function(M)


if __name__ == "__main__":
    os.getcwd()
    print(os.getcwd())
    data = dadi.Spectrum.from_file('fs_data.fs')#(os.path.join(os.getcwd(), '/adjoint_state_method/fs_data.fs'))
    ns = data.sample_sizes
    print("ns", ns)
    xx = dadi.Numerics.default_grid(ns[0])
    #xx = simple_grid(ns)
    nu = 0.01 #population size 0.01, 1.0, 0.005, 0.05
    gamma = 1 #Selection coefficient
    """The selection coefficient (s) of a given genotype as related to the fitness or
     adaptive value (W) of that genotype is defined as s = 1 - W. (Fitness is the
      relative probability that a genotype will reproduce.)"""
    h = 0.005  # dominance coefficient
    beta = 0.05 #Breeding ratio (The sex ratio is the ratio of males to females in a population)
    """beta: Breeding ratio, beta = Nf / Nm.
    alpha: Male to female mutation rate ratio, beta = mu_m / mu_f."""
    initial_phi = dadi.PhiManip.phi_1D(xx, nu, theta0=1.0, gamma=1, h=0.005, theta=None, beta=0.05)

    try:
        main(data, ns, xx, nu, gamma, h, beta, initial_phi)
    except:
        logging.exception("Unexpected error")
"""
pts = 3
xx = simple_grid(pts)
initial_phi = np.array([0.3, 0.4, 0.5])
ns = len(initial_phi)

nu = 1
gamma = 5
h = 0.5 #dominance
beta = 1
M = _Mfunc1D(xx, gamma, h)
MInt = _Mfunc1D((xx[:-1] + xx[1:]) / 2, gamma, h)
dMdgamma = _Mfunc1D_dgamma((xx[:-1] + xx[1:]) / 2, h) # Int
dMdh = _Mfunc1D_dh((xx[:-1] + xx[1:]) / 2, gamma) #Int
print("M", M, "dMdgamma", dMdgamma, "dMdh", dMdh)
V = _Vfunc(xx, nu, beta)
#VInt = _Vfunc((xx[:-1] + xx[1:]) / 2, nu, beta=beta)
dVdnu = _Vfunc_dnu(xx, nu, beta)
dVdbeta = _Vfunc_dnu(xx, nu, beta)
print("V", V, "dVdnu", dVdnu, "dVdbeta", dVdbeta)
dx = np.diff(xx)
dfactor = _compute_dfactor(dx)
#delj = _compute_delj(dx, M, V) #MInt, VInt)

tridiag = calc_tridiag_matrix(dfactor, MInt, M, V, dx)
print("dfactor", dfactor, "tridiag", tridiag, "determinant", np.linalg.det(tridiag))
inverse_tridiag = calc_inverse_tridiag_matrix(tridiag)
print("inverse_tridiag", inverse_tridiag)
dtridiag_dnu = calc_dtridiag_dnu(dfactor, dVdnu, dx, M)
dtridiag_dbeta = calc_dtridiag_dbeta(dfactor, dVdbeta, dx)
print("dtridiag_dnu", dtridiag_dnu, "determinant", np.linalg.det(dtridiag_dnu), "dtridiag_dbeta",
      dtridiag_dbeta, "determinant", np.linalg.det(dtridiag_dbeta))
dtridiag_dgamma = calc_dtridiag_dgamma(dfactor, dMdgamma, M)
dtridiag_dh = calc_dtridiag_dh(dfactor, dMdgamma, M)
print("dtridiag_dgamma", dtridiag_dgamma, "determinant", np.linalg.det(dtridiag_dgamma),
      "dtridiag_dh", dtridiag_dh, "determinant", np.linalg.det(dtridiag_dh))
dtridiagdTetta = get_dtridiagdTetta(dtridiag_dnu, dtridiag_dbeta, dtridiag_dgamma, dtridiag_dh)
print("dtridiagdTetta", dtridiagdTetta, "determinant", np.linalg.det(dtridiagdTetta), dtridiagdTetta.shape)

dtridiag_inverse_dTetta = get_dtridiag_inverse_dTetta(inverse_tridiag, dtridiagdTetta)
print("dtridiag_inverse_dTetta", dtridiag_inverse_dTetta, dtridiagdTetta.shape)
"""
"""
model_AFS = calc_model_AFS(initial_phi, xx, ns)
derivative_model_AFS = calc_dmodel_AFS_dphi(xx, ns)
print("model_AFS", model_AFS)
print("derivative_model_AFS", derivative_model_AFS)

obj_func = S * np.log(model_AFS) - model_AFS - np.log(S)

print("obj_func", obj_func)
dobj_func_dphi = S * np.log(derivative_model_AFS) - derivative_model_AFS - np.log(S)
print("dobj_func_dphi", dobj_func_dphi)
eps = np.sqrt(np.finfo(float).eps)
# derivative_model_AFS_sci = np.gradient(model_AFS, initial_phi)
# print(derivative_model_AFS_sci)

tetta = [nu, gamma, h, beta]

"""
"""
a = np.zeros(initial_phi.shape)
a[1:] += dfactor[1:] * (-MInt * delj - V[:-1] / (2 * dx))

print("grad a gamma", np.gradient(a, gamma))
print("grad a h", np.gradient(a, h))
print("grad a nu", np.gradient(a, nu))
print("grad a beta", np.gradient(a, beta))

c = np.zeros(initial_phi.shape)
c[:-1] += -dfactor[:-1] * (-MInt * (1 - delj) + V[1:] / (2 * dx))
print("grad c gamma", np.gradient(c, gamma))
print("grad c h", np.gradient(c, h))
print("grad c nu", np.gradient(c, nu))
print("grad c beta", np.gradient(c, beta))

b = np.zeros(initial_phi.shape)
b[:-1] += -dfactor[:-1] * (-MInt * delj - V[:-1] / (2 * dx))
b[1:] += dfactor[1:] * (-MInt * (1 - delj) + V[1:] / (2 * dx))
print("grad b gamma", np.gradient(b, gamma))
print("grad b h", np.gradient(b, h))
print("grad b nu", np.gradient(b, nu))
print("grad b beta", np.gradient(b, beta))

print("a, b, c", a, b, c, len(a), len(b), len(c))
A_matrix = scipy.sparse.diags([a[1:], b, -c[:-1]], [-1, 0, 1]).toarray()
print("A_matrix to array", A_matrix)
print("grad gamma", np.gradient(A_matrix, gamma))
grad_h_a = np.gradient(A_matrix, h)
print("grad h", grad_h_a, type(grad_h_a), len(grad_h_a))

A_matrix_inverse = scipy.linalg.inv(A_matrix)
print("A_matrix", A_matrix, type(A_matrix))

dAdgamma = scipy.sparse.dia_matrix(
    scipy.sparse.diags([np.gradient(a[1:], gamma), np.gradient(b, gamma), np.gradient(-c[:-1], gamma)],
                       [-1, 0, 1]))
dAdh = scipy.sparse.dia_matrix(scipy.sparse.diags([np.gradient(a[1:], h), np.gradient(b, h), np.gradient(-c[:-1], h)],
                                                  [-1, 0, 1]))
dAdnu = scipy.sparse.dia_matrix(
    scipy.sparse.diags([np.gradient(a[1:], nu), np.gradient(b, nu), np.gradient(-c[:-1], nu)],
                       [-1, 0, 1]))
dAdbeta = scipy.sparse.dia_matrix(
    scipy.sparse.diags([np.gradient(a[1:], beta), np.gradient(b, beta), np.gradient(-c[:-1], beta)],
                       [-1, 0, 1]))
# diadA_matrixdgamma = scipy.sparse.dia_matrix(dAdgamma)
# diadA_matrixdh = scipy.sparse.dia_matrix(dAdh)
# diadA_matrixdh = scipy.sparse.dia_matrix(dAdnu)
# diadA_matrixdh = scipy.sparse.dia_matrix(dAdbeta)
# print("dA_matrixdgamma", dAdgamma)
# print("dA_matrixdh", dAdh)
# print("dA_matrixdnu", dAdnu)
# print("dA_matrixdbeta", dAdbeta)
dAdtetta = scipy.sparse.dia_matrix.multiply(dAdgamma, dAdh)
dAdtetta = scipy.sparse.dia_matrix.multiply(dAdtetta, dAdnu)
dAdtetta = scipy.sparse.dia_matrix.multiply(dAdtetta, dAdbeta)
print("dAdtetta", dAdtetta.toarray())

dia_mult [[8.27314515e+00 2.89696243e-01 0.00000000e+00 0.00000000e+00
  0.00000000e+00]
 [2.15960854e+01 7.24956315e-01 8.13184324e-02 0.00000000e+00
  0.00000000e+00]
 [0.00000000e+00 2.40295244e+00 4.88281250e-01 2.80166616e-02
  0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 3.40345646e-02 3.23888266e+01
  5.48197611e-02]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 7.57744064e-02
  7.57187304e+01]]

# matrix_sparse = sparse.csr_matrix(matrix)
print("initial_phi", initial_phi)
next_phi = calc_next_phi(initial_phi, A_matrix, 0.1, xx, 30)
print("next_phi", next_phi)
# functional_F = functional_F(initial_phi, next_phi) #, A_matrix)
# print("functional_F", functional_F)
T = 3
dFdphi = np.ones(ns)
print("dFdphi", dFdphi)
dFdTetta = dFdTetta(A_matrix, T, dAdtetta)

adjoint_sol = scipy.sparse.dia_matrix(np.full(ns, np.matmul(dFdphi.T, derivative_model_AFS)))
# adjoint_sol = np.matmul(dFdphi.T, derivative_model_AFS) #np.linalg.solve(A_matrix.T, first_der)
print("adjoint_sol", adjoint_sol)

phi_final = feedforward(M, initial_phi, A_matrix, xx, T, nu=1, gamma=0, h=0.5, theta0=1,
                        initial_t=0, beta=1)
print("phi_final", phi_final)
# adjoint_sol = scipy.sparse.dia_matrix(adjoint_sol)
target_grad = scipy.sparse.dia_matrix.multiply(adjoint_sol, dFdTetta)  # np.matmul(adjoint_sol, dFdTetta)
print("target_grad", target_grad.toarray())
"""
"""
def ascent(xsol, params, vcb, vvb, alpha=1, max=50, savefig=False):
    # Applies gradient ascent to find optimal parameters using a learning rate alpha.
    s = copy.copy(xsol)
    p = copy.copy(params)
    oldg = 0.
    newg = 1.
    cacheg = 0
    cachep = np.zeros(len(p))
    gs = []
    for i in range(max):
        oldg = g(s)
        grad = dgdp(s, p, vcb, vvb)
        p = p + alpha * grad
        s = solve(s, p, vcb, vvb,
                  quiet=False)  # solve(np.concatenate((np.full(N,1/math.sqrt(N)),np.full(N,1/math.sqrt(N)),np.ones(N),(10**-15)*np.ones(2))),p,quiet=True) # solve(s,p,quiet=True)
        newg = g(s)
        gs.append(newg)
        print("newg:", newg)
        print("oldg:", oldg)
        print("difference:", newg - oldg)
        dv = np.concatenate((np.zeros(barrier_locs[0] + 1), p, np.zeros(N - barrier_locs[1] - 1)))
        if newg > cacheg:
            cacheg = copy.copy(newg)
            cachep = copy.deepcopy(p)
        print("\n")
        if savefig is True:
            plt.figure(1, dpi=120)
            plt.plot(-vvb + dv)
            plt.xlabel("Lattice label")
            plt.ylabel("Valence band (J)")
            plt.title("Overlap: " + '%.10f' % newg)
            plt.savefig('images/' + str(i) + '.png', dpi=300)
    # plt.figure(1)
    # plt.xlabel("Lattice label")
    # plt.ylabel("Probability density")
    # plt.show()
    print(gs)
    return cachep
    
def dFdTetta(A_matrix, T, dAdtetta):
    inv_matrix = scipy.linalg.inv(A_matrix)
    dia_inverse = scipy.sparse.dia_matrix(inv_matrix)
    matrix_power = scipy.sparse.dia_matrix.power(dia_inverse, T)
    res = scipy.sparse.dia_matrix.multiply(matrix_power, dAdtetta)
    res = scipy.sparse.dia_matrix.multiply(res, dia_inverse)
    return res


nu=1
gamma=5
h=0.5
beta=1
A_matrix [[ 2.1875   0.8125   0.       0.       0.     ]
 [-1.09375  4.25    -0.34375  0.       0.     ]
 [ 0.      -3.84375  4.      -0.84375  0.     ]
 [ 0.       0.      -4.34375  1.75    -1.09375]
 [ 0.       0.       0.      -5.1875  -2.1875 ]]

"""
