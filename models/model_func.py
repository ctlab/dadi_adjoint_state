import dadi


def model_func(params, ns, pts):
    """
    :param nu: population size
    :param gamma: Selection coefficient
    The selection coefficient (s) of a given genotype as related to the fitness or
    adaptive value (W) of that genotype is defined as s = 1 - W. (Fitness is the
    relative probability that a genotype will reproduce.)
    :param h: dominance coefficient
    :param beta: breeding ratio (The sex ratio is the ratio of males to females in a population)
    beta: Breeding ratio, beta = Nf / Nm.
    ns: Sequence of P sample sizes for each population.
    """
    nu, gamma, h, beta, theta0 = params
    T = 3
    xx = dadi.Numerics.default_grid(pts)
    phi = dadi.PhiManip.phi_1D(xx, nu=nu, theta0=theta0, gamma=gamma,
                                                       h=h, theta=None,
                                                       beta=beta)
    phi = dadi.Integration.one_pop(phi, xx, T, nu=nu, gamma=gamma, h=h, theta0=theta0, initial_t=0, frozen=False,
                                   beta=beta)
    fs = dadi.Spectrum.from_phi(phi, ns, (xx,))
    return fs