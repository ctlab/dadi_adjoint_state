"""
Single population demographic models.
"""
import dadi
import numpy
import torch
from dadi.Spectrum_mod import Spectrum
from sklearn.preprocessing import MinMaxScaler
import Integration
import PhiManip


def snm(notused, ns, pts):
    """
    Standard neutral model.

    ns = (n1,)

    n1: Number of samples in resulting Spectrum_mod.py
    pts: Number of grid points to use in integration.
    """
    xx = dadi.Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)

    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs


def scaler(data):
    data = numpy.asarray([data])
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


def torch_scaler(tensor):
    # scale = 1.0 / (tensor.max(dim=1, keepdim=True)[0] - tensor.min(dim=1, keepdim=True)[0])
    # tensor.mul_(scale).sub_(tensor.min(dim=1, keepdim=True)[0])
    scale = 1.0 / (tensor.max() - tensor.min())
    tensor.mul_(scale).sub_(tensor.min())
    return tensor


# def two_epoch_ASM(params, T, initial_t, ns, pts, xx=Numerics.default_grid):
#     """
#     Instantaneous size change some time ago.
#
#     params = (nu,T)
#     ns = (n1,)
#
#     nu: Ratio of contemporary to ancient population size
#     T: Time in the past at which size change happened (in units of 2*Na
#        generations) [T = t/2*Nref]
#     n1: Number of samples in resulting Spectrum_mod.py
#     pts: Number of grid points to use in integration.
#     """
#     nu = params[0]
#     # T = params[-1]
#     # initial_t = params[-2]
#     theta0, gamma, h, theta, beta = torch.tensor(1.0), torch.tensor(0), torch.tensor(0.5), None, torch.tensor(1)
#     # final_t = T
#     xx = torch.as_tensor(xx(pts))
#     # xx = Numerics.default_grid(pts)
#     phi = dadi_torch.PhiManip.phi_1D_genic(xx, nu=nu, theta0=theta0, gamma=gamma, theta=theta, beta=beta)
#     # print("phi.phi_1D_genic={}".format(phi))
#     # phi = PhiManip.phi_1D(xx, nu=nu, gamma=gamma, h=h, theta=theta, beta=beta)
#     print("phi size", phi.shape)
#     # phi_initial = torch_scaler(phi)
#     phi_initial = phi
#     # phi = Integration.one_pop(phi, xx, T, nu=nu, gamma=gamma, h=h, theta0=theta0, initial_t=initial_t, beta=1)
#     phi, phi_inj = dadi_torch.Integration._one_pop_const_params(phi, xx, T, nu=nu, gamma=gamma,
#                                                                h=h, theta0=theta0, initial_t=initial_t, beta=beta)
#     #print("phi", phi, "\nns", ns, "xx", xx.shape)
#     fs = Spectrum.from_phi(phi.detach().numpy(), ns, (xx.detach().numpy(),), mask_corners=True, force_direct=True)
#     return fs, phi_initial, phi, phi_inj, xx


def two_epoch(params, ns, pts):
    """
    Instantaneous size change some time ago.

    params = (nu,T)
    ns = (n1,)

    nu: Ratio of contemporary to ancient population size
    T: Time in the past at which size change happened (in units of 2*Na 
       generations) 
    n1: Number of samples in resulting Spectrum_mod.py
    pts: Number of grid points to use in integration.
    """
    print("params", params)
    nu, T = params[0][0], params[1][0]
    print("params two_epoch: nu {}, T {}, type {}, ndim {}".format(nu, T, type(nu), nu.ndim))
    xx = dadi.Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(torch.as_tensor(xx), nu=nu, h=torch.tensor(0.6))
    # phi = dadi_torch.PhiManip.phi_1D_genic(torch.as_tensor(xx), nu=nu)
    # print("phi INITial from two_epoch PhiManip.phi_1D_genic", phi)
    # phi = dadi.Integration.one_pop(phi, xx, T, nu)
    phi, _ = Integration._one_pop_const_params(phi, torch.as_tensor(xx), T, nu=nu)
    # print("phi from two_epoch Integration.one_pop", phi)
    fs = Spectrum.from_phi(phi.detach().numpy(), ns, (xx,))
    print("fs={}".format(fs))
    return fs


def growth(params, ns, pts):
    """
    Exponential growth beginning some time ago.

    params = (nu,T)
    ns = (n1,)

    nu: Ratio of contemporary to ancient population size
    T: Time in the past at which growth began (in units of 2*Na 
       generations) 
    n1: Number of samples in resulting Spectrum_mod.py
    pts: Number of grid points to use in integration.
    """
    nu, T = params

    xx = dadi.Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)

    nu_func = lambda t: numpy.exp(numpy.log(nu) * t / T)
    phi = Integration.one_pop(phi, xx, T, nu_func)

    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs


def bottlegrowth(params, ns, pts):
    """
    Instantanous size change followed by exponential growth.

    params = (nuB,nuF,T)
    ns = (n1,)

    nuB: Ratio of population size after instantanous change to ancient
         population size
    nuF: Ratio of contemporary to ancient population size
    T: Time in the past at which instantaneous change happened and growth began
       (in units of 2*Na generations) 
    n1: Number of samples in resulting Spectrum_mod.py
    pts: Number of grid points to use in integration.
    """
    nuB, nuF, T = params

    xx = dadi.Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)

    nu_func = lambda t: nuB * numpy.exp(numpy.log(nuF / nuB) * t / T)
    phi = Integration.one_pop(phi, xx, T, nu_func)

    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs


def three_epoch(params, ns, pts):
    """
    params = (nuB,nuF,TB,TF)
    ns = (n1,)

    nuB: Ratio of bottleneck population size to ancient pop size
    nuF: Ratio of contemporary to ancient pop size
    TB: Length of bottleneck (in units of 2*Na generations) 
    TF: Time since bottleneck recovery (in units of 2*Na generations) 

    n1: Number of samples in resulting Spectrum_mod.py
    pts: Number of grid points to use in integration.
    """
    nuB, nuF, TB, TF = params

    xx = dadi.Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)

    phi = Integration.one_pop(phi, xx, TB, nuB)
    phi = Integration.one_pop(phi, xx, TF, nuF)

    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs


def three_epoch_ASM(params, ns, pts, xx=dadi.Numerics.default_grid, initial_t=0):
    """
    params = (nuB,nuF,TB,TF)
    ns = (n1,)

    nuB: Ratio of bottleneck population size to ancient pop size
    nuF: Ratio of contemporary to ancient pop size
    TB: Length of bottleneck (in units of 2*Na generations) (t/2*Nref)
    TF: Time since bottleneck recovery (in units of 2*Na generations)

    n1: Number of samples in resulting Spectrum_mod.py
    pts: Number of grid points to use in integration.
    """
    nuB, nuF, TB, TF = params
    xx = xx(pts)
    phi = PhiManip.phi_1D(xx)

    phi = Integration.one_pop(phi, xx, TB, nuB, initial_t=initial_t)
    phi = Integration.one_pop(phi, xx, TF, nuF, initial_t=initial_t)

    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs
