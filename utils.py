import numpy as np

def plot_distribution(dist, ax):
    xmin, xmax = dist.ppf([0.0001, 0.9999])
    x = np.linspace(xmin, xmax, 1000)
    y = dist.pdf(x)

    ax.set_xlabel("x value")
    ax.set_ylabel("likelihood")
    ax.plot(x, y)
    return ax
