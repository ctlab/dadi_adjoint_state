import models.model_func as model_func
import dadi

n_pop = 1
pop_labels = ["Pop 1"]

par_labels = ['nu', 'gamma', 'h', 'beta']
popt = [0.01, 1.0, 0.005, 0.05]

lower_bound = [1e-3, 1e-3, 0, 0]
upper_bound = [100, 100, 5, 5]

ns_per_pop = 20
ns = [ns_per_pop for _ in range(n_pop)]
print("ns", ns)

mu = 2.5e-8  # mutation rate
L = 20000000  # effective length of sequence
Nanc = 100
theta = 4 * mu * L * Nanc  # mutation flux
print("theta", theta)
# Get maximum log-likelihood
model = model_func.model_func(popt, ns_per_pop)
data = model * theta
max_ll = dadi.Inference.ll_multinom(model, data)
print("model", model)
print("data", data)
print('Maximum log composite likelihood: {0}'.format(max_ll))

if __name__ == "__main__":
    data.to_file('fs_data.fs')
    print('Simulated data saved to fs_data.fs')

    theta = dadi.Inference.optimal_sfs_scaling(model, data)
    print('Optimal value of theta: {0}'.format(theta))
    theta0 = 4 * mu * L
    Nanc = int(theta / theta0)
    print('Size of the ancestral population: {0}'.format(Nanc))

    dadi.Plotting.plot_1d_fs(model, show=True)

    dadi.Plotting.plot_1d_comp_Poisson(model, data, fig_num=None, residual='Anscombe',
                            plot_masked=False, show=True)

"""
def model1((nu, t), (n1, ), (g, h), pts):
    xx = dadi.Numerics.default_grid(pts)
    phi = 0.0*dadi.PhiManip.phi_1D(xx)
    phi = dadi.Integration.one_pop(phi, xx, t, nu=nu, gamma=g, h=h)
    sfs = dadi.Spectrum.from_phi(phi, (n1, ), (xx, ))
    return sfs 
"""