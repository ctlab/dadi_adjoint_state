import os
from matplotlib import pyplot as plt
import numpy as np

maxeval = 500
y = range(-300, -31)
data_ll = np.random.uniform(-300, -31, 500)
data_ll_sort = sorted(data_ll)
opt2color = ["tab:blue", "tab:orange", "tab:purple"]


def draw_comparison(n_samples, ymin=None):
    fig, ax = plt.subplots(figsize=(7, 7))
    # read data

    x_points = range(maxeval)
    x_points = np.array(x_points)
    y_points = np.array(data_ll_sort)
    print(type(x_points), x_points.shape)
    print(type(y_points), y_points.shape)

    x_points.reshape([1, 500])
    y_points.reshape([1, 500])
    print(type(x_points), x_points.shape)
    print(type(y_points), y_points.shape)
    # Show empirical quantiles
    y_points = np.cumsum(-y_points)
    ax.fill_between(x_points,
                    *np.quantile(y_points, q=(0.25, 0.75), axis=0),
                    color=opt2color[0],
                    alpha=0.15)

       # Plot some sample paths
    # for y in y_points[:n_samples]:
    #     ax.plot(x_points, y, alpha=0.5, linewidth=0.5, color=opt2color[0])

        # Plot median performance
    # ax.plot(x_points, *np.quantile(y_points, q=0.5, axis=0),
    #         alpha=0.8, linewidth=0.75, color=opt2color[0], label='haha')
    ax.plot(x_points, y_points, alpha=0.5, linewidth=0.5, color=opt2color[0], label='haha')

    plt.legend(loc=0)
    plt.show()

draw_comparison(5)
