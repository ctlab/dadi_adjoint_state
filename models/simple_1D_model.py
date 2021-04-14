import dadi


def simple_1D_model_func(params, ns, pts, xx=dadi.Numerics.default_grid, initial_t=0, final_t=3):
    """
    :param params:
    nu: population size
    gamma: Selection coefficient
    The selection coefficient (s) of a given genotype as related to the fitness or
    adaptive value (W) of that genotype is defined as s = 1 - W. (Fitness is the
    relative probability that a genotype will reproduce.)
    h: dominance coefficient
    beta: breeding ratio (The sex ratio is the ratio of males to females in a population), beta = Nf / Nm.
    :param ns: Sequence of P sample sizes for each population
    :param pts: list of grid points to use in evaluation.
    :param xx: grid
    :param initial_t:
    :param final_t:

    from dadi.Inference.optimize:
        func_args: Additional arguments to model_func. It is assumed that
               model_func's first argument is an array of parameters to
               optimize, that its second argument is an array of sample sizes
               for the sfs, and that its last argument is the list of grid
               points to use in evaluation.
               Using func_args.
               For example, you could define your model function as
               def func((p1,p2), ns, f1, f2, pts):
                   ....
               If you wanted to fix f1=0.1 and f2=0.2 in the optimization, you
               would pass func_args = [0.1,0.2] (and ignore the fixed_params
               argument).
    func_kwargs: Additional keyword arguments to model_func.
    """
    nu, gamma, h, beta, theta0 = params
    xx = xx(pts)
    phi = dadi.PhiManip.phi_1D(xx, nu=nu, theta0=theta0, gamma=gamma, h=h, theta=None, beta=beta)
    phi = dadi.Integration.one_pop(phi, xx, final_t, nu=nu, gamma=gamma, h=h, theta0=theta0, initial_t=initial_t,
                                   frozen=False, beta=beta)
    fs = dadi.Spectrum.from_phi(phi, ns, (xx,))
    return fs
