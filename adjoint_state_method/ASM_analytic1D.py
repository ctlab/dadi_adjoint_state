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
use_delj_trick = False


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
    return (- x * (1 - x) * (beta + 1.) ** 2) / (nu ** 2 * 4. * beta)


def _Vfunc_dbeta(x, nu, beta=1):
    return (2 * x * (1 - x) * (beta + 1.) * 4 * nu * beta - 4 * nu * x * (1 - x) * (beta + 1) ** 2) / (
                16 * nu ** 2 * beta ** 2)


def _Mfunc1D_dgamma(x, h):
    return 2 * x * (h + (1 - 2 * h) * x) * (1 - x)


def _Mfunc1D_dh(x, gamma):
    return 2 * gamma * x * (2 * x ** 2 - 3 * x + 1)


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
        # wj = 2 *MInt*dx[upslice]
        wj = MInt * dx[upslice]
        epsj = np.exp(wj / VInt[upslice])
        delj = 1 / wj - 1 / (epsj - 1)  # (-epsj*wj + epsj * VInt[upslice] - VInt[upslice])/(wj - epsj*wj)
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
    maxVM = max(0.25 / nu, sum(ms),
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


def _inject_mutations_1D(phi, dt, xx, theta0):
    # Inject novel mutations for a timestep.
    # xx = dadi.Numerics.default_grid(pts)
    phi[1] += dt / xx[1] * theta0 / 2 * 2 / (xx[2] - xx[0])
    return phi


def calc_injected_and_next_phi(previous_phi, tridiag_matrix, this_dt, xx, theta0):
    injected_phi = _inject_mutations_1D(previous_phi, this_dt, xx, theta0)
    next_phi = np.matmul(np.linalg.inv(tridiag_matrix), injected_phi)  # scipy.linalg.inv(tridiag_matrix)
    return injected_phi, next_phi


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


def calc_dtridiag_dnu(initial_phi, dfactor, dVdnu, dx, nu, M):
    a = np.zeros(initial_phi.shape)
    a[1:] += -dfactor[1:] * dVdnu[:-1] / (2 * dx)
    c = np.zeros(initial_phi.shape)
    c[:-1] += -dfactor[:-1] * dVdnu[1:] / (2 * dx)
    b = np.zeros(initial_phi.shape)
    b[:-1] += dfactor[:-1] * dVdnu[:-1] / (2 * dx)
    b[1:] += dfactor[1:] * dVdnu[1:] / (2 * dx)

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


def get_dtridiag_dtheta(dtridiag_dnu, dtridiag_dbeta, dtridiag_dgamma, dtridiag_dh):
    return np.array([dtridiag_dnu, dtridiag_dbeta, dtridiag_dgamma, dtridiag_dh])


def get_dtridiag_inverse_dtheta(inverse_tridiag, dtridiag_dtheta):
    """
    (dA)^-1/dTheta = -A^(-1)(dA/dTheta)A^(-1)
    """
    result = np.matmul(-inverse_tridiag, dtridiag_dtheta)
    result = np.matmul(result, inverse_tridiag)
    return result


def feedforward(theta, previous_phi, tridiag_matrix, inverse_tridiag, xx, ns, dx, dfactor, T=3, theta0=1, initial_t=0,
                delj=0.5):
    nu, gamma, h, beta = theta
    M = _Mfunc1D(xx, gamma, h)
    # MInt = _Mfunc1D((xx[:-1] + xx[1:]) / 2, gamma, h)
    dM_dgamma_Int = _Mfunc1D_dgamma((xx[:-1] + xx[1:]) / 2, h)
    dM_dh_Int = _Mfunc1D_dh((xx[:-1] + xx[1:]) / 2, gamma)
    dV_dnu = _Vfunc_dnu(xx, nu, beta)
    dV_dbeta = _Vfunc_dbeta(xx, nu, beta)
    dtridiag_dnu = calc_dtridiag_dnu(previous_phi, dfactor, dV_dnu, dx, nu, M)
    dtridiag_dbeta = calc_dtridiag_dbeta(previous_phi, dfactor, dV_dbeta, dx)
    dtridiag_dgamma = calc_dtridiag_dgamma(previous_phi, dfactor, dM_dgamma_Int, M, dx, delj)
    dtridiag_dh = calc_dtridiag_dh(previous_phi, dfactor, dM_dh_Int, M, dx, delj)
    dtridiag_dTheta = get_dtridiag_dtheta(dtridiag_dnu, dtridiag_dbeta, dtridiag_dgamma, dtridiag_dh)
    dtridiag_inverse_dTheta = get_dtridiag_inverse_dtheta(inverse_tridiag, dtridiag_dTheta)
    dF_dtheta = np.zeros([dtridiag_inverse_dTheta.shape[0], ns[0]], dtype=float)

    dt = _compute_dt(dx, nu, [0], gamma, h)
    current_t = initial_t
    while current_t < T:
        this_dt = min(dt, T - current_t)
        phi_injected, phi = calc_injected_and_next_phi(previous_phi, tridiag_matrix, this_dt, xx, theta0)
        previous_phi = phi
        current_t += this_dt
        dF_dtheta += np.matmul(dtridiag_inverse_dTheta, phi_injected)
        dF_dtheta = np.matmul(dF_dtheta, inverse_tridiag)
    # inject
    dF_dtheta = np.matmul(dF_dtheta, tridiag_matrix)

    if not phi.any():
        logging.error('Phi not counted')
        return
    return phi, dF_dtheta


def calc_target_grad(dF_dtheta, adjoint_field):
    target_grad = np.matmul(dF_dtheta, adjoint_field)
    # logging.debug("target_grad: {0}".format(target_grad))
    norm = np.linalg.norm(target_grad)
    # logging.debug("target_grad norm: {0}".format(norm))
    return target_grad


def _from_phi_1D_direct(phi, n, xx, mask_corners=True,
                        het_ascertained=None):
    """
    Compute sample Spectrum from population frequency distribution phi.
    ns: Sequence of P sample sizes for each population.
    xx: Sequence of P one-dimensional grids on which phi is defined.
    See from_phi for explanation of arguments.
    """
    n = round(n)
    data = np.zeros(n + 1)
    for ii in range(0, n + 1):
        factorx = scipy.special.comb(n, ii) * xx ** ii * (1 - xx) ** (n - ii)
        if het_ascertained == 'xx':
            factorx *= xx * (1 - xx)
        data[ii] = trapz(factorx * phi, xx)
    return dadi.Spectrum(data, mask_corners=mask_corners)


def _from_phi_1D_direct_dphi_analytical(n, xx, dfactor, mask_corners=True,
                             het_ascertained=None):
    """
    Compute sample Spectrum from population frequency distribution phi.
    See from_phi for explanation of arguments.
    """
    """ test failed """
    n = round(n)
    delta_dfactor = np.diff(dfactor)
    double_delta_xx = np.diff(xx, 2)
    data = np.zeros(n)
    for ii in range(0, n):
        factorx = scipy.special.comb(n, ii) * xx ** ii * (1 - xx) ** (n - ii)
        if het_ascertained == 'xx':
            factorx *= xx * (1 - xx)
        # data[ii] = trapz(factorx, double_delta_xx/2)
        # data[ii] *= double_delta_xx/2
        data[ii] *= delta_dfactor / 2
    return dadi.Spectrum(data, mask_corners=mask_corners)


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


def calc_objective_func(phi, xx, ns, observed_spectrum):
    """ objective_func = ll"""
    model = _from_phi_1D_direct(phi, ns, xx)
    obj_func = observed_spectrum * np.log(model) - model - np.log(observed_spectrum)
    return obj_func


def dll_dphi(model, observed_spectrum, ns, xx):
    """ analytical derivative"""
    dmodel_dphi = _from_phi_1D_direct_dphi_directly(ns[0], xx)
    return dmodel_dphi * (observed_spectrum/model - 1)


def calc_objective_func_from_theta(theta, phi_initial, tridiag, inverse_tridiag, xx, ns, dx, dfactor, observed_spectrum,
                                   T, theta0, initial_t, delj):
    phi, _ = feedforward(theta, phi_initial, tridiag, inverse_tridiag, xx, ns, dx, dfactor, T=T, theta0=theta0,
                         initial_t=initial_t, delj=delj)
    model = _from_phi_1D_direct(phi, ns[0], xx)
    obj_func = observed_spectrum * np.log(model) - model - np.log(observed_spectrum)
    return obj_func


"""
def functional_F(current_phi, next_phi):
    return next_phi - current_phi
"""


def dfunctionalF_dphi(ns):
    """ ns - size of grid"""
    return -1 * np.ones(ns)


def ascent(theta, dF_dtheta, adjoint_field, phi, tridiag, inverse_tridiag, xx, ns, dx, dfactor, T, theta0=1,
           initial_t=0):
    learning_rate = 0.1
    target_grad = calc_target_grad(dF_dtheta, adjoint_field)
    for i in range(10):
        # while np.linalg.norm(target_grad) >= 0.5:
        theta_optimized = theta + learning_rate * target_grad
        theta = theta_optimized
        phi, dF_dtheta = feedforward(theta, phi, tridiag, inverse_tridiag, xx, ns, dx, dfactor, T=T, theta0=1,
                                     initial_t=0)
        target_grad = calc_target_grad(dF_dtheta, adjoint_field)
        # logging.debug("theta: {0}".format(theta))
    return theta


def plot_function(func_output):
    plt.figure(figsize=(10, 7))
    x = np.linspace(-1, 2, len(func_output))
    y = func_output
    plt.plot(x, y)
    plt.ylabel("Y")
    plt.xlabel("X")
    # plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    plt.scatter(x, y, linewidths=5)
    # plt.scatter([3/8], [F(3/8)], lw=5)
    plt.show()


def main(observed_spectrum, ns, xx, nu, gamma, h, beta, initial_phi, T):
    logger = logging.getLogger()
    logger.setLevel(10)
    # logging.basicConfig(format='[%(asctime)s] %(levelname).1s %(message)s', level=logging.DEBUG)

    logger.debug("observed spectrum: {0}\n xx: {1}\n nu: {2}\n gamma: {3}\n"
                 " h: {4}\n beta: {5}\n initial_phi: {6}\n".format(observed_spectrum, xx, nu, gamma, h, beta,
                                                                   initial_phi))

    theta = np.array([nu, gamma, h, beta])
    M = _Mfunc1D(xx, gamma, h)
    MInt = _Mfunc1D((xx[:-1] + xx[1:]) / 2, gamma, h)
    dM_dgamma = _Mfunc1D_dgamma((xx[:-1] + xx[1:]) / 2, h)  # Int
    dM_dh = _Mfunc1D_dh((xx[:-1] + xx[1:]) / 2, gamma)  # Int
    logging.debug("M: {0}\n MInt: {1}\n dM_dgamma: {2}\n dM_dh: {3}".format(M, MInt, dM_dgamma, dM_dh))

    V = _Vfunc(xx, nu, beta)
    VInt = _Vfunc((xx[:-1] + xx[1:]) / 2, nu, beta=beta)
    dV_dnu = _Vfunc_dnu(xx, nu, beta)
    dV_dbeta = _Vfunc_dbeta(xx, nu, beta)
    logging.debug("V: {0} \n dV_dnu: {1} \n dV_dbeta: {2} \n".format(V, dV_dnu, dV_dbeta))

    dx = np.diff(xx)
    dfactor = _compute_dfactor(dx)
    delj = _compute_delj(dx, MInt, VInt)
    logging.debug("delj: {0}".format(delj))
    tridiag = calc_tridiag_matrix(initial_phi, dfactor, MInt, M, V, dx, delj)
    logging.debug("dfactor: {0}\n tridiag: \n {1} determinant: {2}".format(dfactor, tridiag, np.linalg.det(tridiag)))
    inverse_tridiag = calc_inverse_tridiag_matrix(tridiag)
    logging.debug("inverse_tridiag: \n {0} \n shape {1} \n".format(inverse_tridiag, inverse_tridiag.shape))

    phi, dF_dtheta = feedforward(theta, initial_phi, tridiag, inverse_tridiag, xx, ns, dx, dfactor, T=T, theta0=1,
                                 initial_t=0, delj=delj)
    logging.debug("phi from feedforward: {0}\n dF_dtheta (third derivative for ASM): \n {1}".format(phi, dF_dtheta))
    model = _from_phi_1D_direct(phi, ns[0], xx)
    logging.debug("AFS _from_phi_1D_direct: {0}\n".format(model))
    dobjective_func_dphi = dll_dphi(model, observed_spectrum, ns, xx)
    logging.debug("dll_dphi (dobjective_func_dphi) - (first derivative for ASM): {0}\n".format(dobjective_func_dphi))
    dF_dphi = dfunctionalF_dphi(ns[0])
    logging.debug("dF_dphi - (second derivative for ASM): {0}\n".format(dF_dphi))

    adjoint_field = np.full(ns, np.matmul(dF_dphi.T, np.asarray(dobjective_func_dphi)[1:]))
    logging.debug("adjoint-state lagrange multipliers: {0}".format(adjoint_field))
    target_grad = calc_target_grad(dF_dtheta, adjoint_field)
    logging.debug("target_grad: {0}\n".format(target_grad))

    objective_functional = calc_objective_func(phi, xx, ns[0], observed_spectrum)
    logging.debug("objective_functional: {0}\n".format(objective_functional))
    # objective_functional_from_theta = calc_objective_func_from_theta(theta, initial_phi, tridiag, inverse_tridiag, xx,
    #                                                                ns, dx, dfactor, observed_spectrum, T=T, theta0=1,
    #                                                                 initial_t=0, delj=delj)
    # logging.debug("objective_functional_from_theta: {0}\n".format(objective_functional_from_theta))
    theta_opt = ascent(theta, dF_dtheta, adjoint_field, phi, tridiag, inverse_tridiag, xx, ns, dx, dfactor, T)
    logging.debug("theta_opt: {0}".format(theta_opt))
    print(len(observed_spectrum))
    opt_res = scipy.optimize.minimize(calc_objective_func_from_theta, theta,
                                      (initial_phi, tridiag, inverse_tridiag, xx, ns, dx, dfactor,
                                       observed_spectrum, T, 1, 0, delj))
    logging.debug("opt_res item: {0}".format(opt_res.item()))
    logging.debug("opt_res: {0}".format(opt_res))


if __name__ == "__main__":
    """
    ns: Sample size of resulting Spectrum
    pts: Number of grid points to use in integration.
    """
    data = dadi.Spectrum.from_file('fs_data.fs')
    ns = data.sample_sizes  # mask corners
    print("ns", ns)
    pts = 19
    xx = dadi.Numerics.default_grid(pts)
    nu = 2  # population size
    gamma = 0.5  # Selection coefficient
    """
    The selection coefficient (s) of a given genotype as related to the fitness or
    adaptive value (W) of that genotype is defined as s = 1 - W. (Fitness is the
    relative probability that a genotype will reproduce.)"""
    h = 0.5  # dominance coefficient
    """
    Genetics. 2011 Feb; 187(2): 553–566.
    doi: 10.1534/genetics.110.124560
        
    We use h to indicate the dominance coefficient of the allele,
    the proportion of fitness effect expressed in the homozygote.
    Thus, a heterozygote would have fitness on average equal to
    1 – hs times the fitness of the wild-type homozygote. If
    h = 0, the heterozygote has fitness equal to that of the 
    wild-type heterozygote, while if h = 1, then the heterozygote 
    has fitness equal to that of the mutant homozygote. In practice 
    h can be negative (indicating overdominance if the mutant allele
    is deleterious), between 0 and 1 as discussed above or even >1
    with underdominance.
    """
    beta = 1  # Breeding ratio (The sex ratio is the ratio of males to females in a population)
    """beta: Breeding ratio, beta = Nf / Nm.
    alpha: Male to female mutation rate ratio, beta = mu_m / mu_f."""
    initial_phi = dadi.PhiManip.phi_1D(xx, nu, theta0=1.0, gamma=1, h=0.5, theta=None, beta=1)
    T = 3

    try:
        main(data, ns, xx, nu, gamma, h, beta, initial_phi, T)
    except:
        logging.exception("Unexpected error")

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
"""
