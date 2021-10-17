import dadi
import torch
import Demographics1D
import os


pop_labels = ["Pop 1"]
par_labels = ['nu', 'gamma', 'h', 'beta', 'theta0']
upper_bound = [100, 1, 1, 10, 1]
lower_bound = [1e-2, 1e-2, 1e-2, 1e-2, 1]
mu = 2.5e-8  # mutation rate
L = 20000000  # effective length of sequence


def sim_growth(ns, pts):
    params = 10, 10  # nu,T
    model = Demographics1D.growth(params, ns, pts)
    Nanc = 100
    theta = 4 * mu * L * Nanc  # mutation flux
    print("theta", theta)
    data = model * theta
    max_ll = dadi.Inference.ll_multinom(model, data)
    print("growth model", model)
    print("data", data)
    print('Maximum log composite likelihood: {}'.format(max_ll))
    return data, model


def sim_two_epoch(ns, pts):
    # params = [0.5, 0.42105263]
    params = [30, 9]
    model = Demographics1D.two_epoch(params, ns, pts)
    Nanc = 100
    theta = 4 * mu * L * Nanc  # mutation flux
    print("theta", theta)
    data = model * theta
    max_ll = dadi.Inference.ll_multinom(model, data)
    print("growth model", model)
    print("data", data)
    print('Maximum log composite likelihood: {}'.format(max_ll))
    return data, model


def sim_two_epoch_ASM(ns, pts):
    params = torch.tensor([8], dtype=torch.float64)  # nu
    T = torch.tensor(8, dtype=torch.float64)
    model, _, _, _, _ = Demographics1D.two_epoch_ASM(params, T, 0, ns, pts, xx=dadi.Numerics.default_grid)
    Nanc = 100
    theta = torch.tensor(4 * mu * L * Nanc, dtype=torch.float64)  # mutation flux
    # print("theta", theta, type(theta))
    print("model", model, type(model))
    data = model * theta
    max_ll = dadi.Inference.ll_multinom(model, data)
    itself = dadi.Inference.ll_multinom(data, data)
    print("two_epoch_ASM model", model)
    print("data", data)
    print("model", model)
    print('Maximum log composite likelihood: {}'.format(max_ll))
    print('likelihood by itself: {}'.format(itself))
    return data, model


def sim_three_epoch_ASM(ns, pts):
    params = [2, 4, 1, 3]  # nuB, nuF, TB, TF
    model = Demographics1D.three_epoch_ASM(params, ns, pts, xx=dadi.Numerics.default_grid, initial_t=0)
    Nanc = 100
    theta = 4 * mu * L * Nanc  # mutation flux
    print("theta", theta)
    data = model * theta
    max_ll = dadi.Inference.ll_multinom(model, data)
    itself = dadi.Inference.ll_multinom(data, data)
    print("two_epoch_ASM model", model)
    print("data", data)
    print('Maximum log composite likelihood: {}'.format(max_ll))
    print('likelihood by itself: {}'.format(itself))
    return data, model


def write_fs(name_of_the_model):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    data.to_file('fs_data_{}.fs'.format(name_of_the_model))
    print('Simulated data saved to fs_data_{}.fs'.format(name_of_the_model))


def calc_Nanc(model, data, name_of_the_model):
    theta = dadi.Inference.optimal_sfs_scaling(model, data)
    print('{}: Optimal value of theta: {}'.format(name_of_the_model, theta))
    theta0 = 4 * mu * L
    Nanc = int(theta / theta0)
    print('{}: Size of the ancestral population: {}'.format(name_of_the_model, Nanc))


def plot(model, data):
    dadi.Plotting.plot_1d_fs(model, show=True)
    dadi.Plotting.plot_1d_comp_Poisson(model, data, fig_num=None, residual='Anscombe',
                                       plot_masked=False, show=True)


if __name__ == "__main__":
    n_pop = 1
    ns_per_pop = 40
    ns = [ns_per_pop for _ in range(n_pop)]
    pts = 30
    print("ns ", ns, "pts ", pts)
    # data, model = sim_growth(ns, pts)
    # write_fs("growth")
    # calc_Nanc(model, data, "growth")

    # data, model = sim_simple_1D_model(ns, pts)
    # write_fs("simple1D")
    # calc_Nanc(model, data, "simple1D")

    # data, model = sim_two_epoch_ASM(ns, pts)  # Maximum log composite likelihood: -70.01700209286719
    # write_fs("two_epoch_ASM")
    # calc_Nanc(model, data, "two_epoch_ASM")

    data, model = sim_two_epoch(ns, pts)
    write_fs("two_epoch")
    calc_Nanc(model, data, "two_epoch")
    plot(model, data)

