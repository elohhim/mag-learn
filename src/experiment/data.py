import itertools
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

plt.style.use('ggplot')
use_contour = False


def get_damped_sine_wave_fun(a=1, lambda_=0.1, omega=1, phi=0):
    """Returns damped sine wave function."""

    def damped_sine_wave(X):
        """exp(-lambda*x") * cos(omega *x * phi)"""
        return a * np.exp(-1 * lambda_ * X) * np.cos(omega * X + phi)

    return damped_sine_wave



def fun_3d(fun):
    def _fun_3d(X):
        x, y = X[:, 0], X[:, 1]
        z = fun(x, y)
        return z.reshape(-1, 1)

    return _fun_3d


def get_half_sphere_fun(r=1):
    """Returns 3d half sphere function."""

    @fun_3d
    def _half_sphere_fun(x, y):
        return np.sqrt(r - np.power(x, 2) - np.power(y, 2))

    return _half_sphere_fun


def get_circular_vibration_fun(a=1, lambda_=0.1, omega=1, phi=0):
    """Return circular vibration function."""

    @fun_3d
    def _circular_vibration_fun(x, y):
        circle = np.sqrt(x ** 2 + y ** 2)
        return a * np.exp(-1 * lambda_ * circle) * np.cos(omega * circle + phi)

    return _circular_vibration_fun


def get_rosenbrock_fun(a=1, b=100):
    """Returns Rosenbrock function."""

    @fun_3d
    def rosenbrock(x, y):
        """(a - x)^2 + b(y - x^2)^2"""
        return (a - x) ** 2 + b * (y - x ** 2) ** 2

    return rosenbrock


def dump_fun_data(Xs, Y, path, name):
    """Dumps function data to file."""
    data = np.append(Xs, Y, axis=1)
    np.savetxt(os.path.join(path, name), data)


def plot_2d_data(X, Y, plot_type):
    """Plots 2D data."""
    calls = {
        'plot': (plt.plot, {}),
        'scatter': (plt.scatter, {'marker': 'o'})
    }
    try:
        call, plot_kwargs = calls[plot_type]
    except KeyError:
        call, plot_kwargs = calls['plot']
    call(X, Y, **plot_kwargs)


def plot_3d_data(X, Y, plot_type, ax=None):
    """Plots 3D data."""
    xs = np.hsplit(X, 2)
    if ax is None:
        ax = plt.axes(projection=None if use_contour else '3d')
    if plot_type is None or plot_type == 'plot':
        flatten_xs = [x.flatten() for x in xs]
        flatten_y = Y.flatten()
        if use_contour:
            ax.tricontour(*flatten_xs, flatten_y, colors='k')
            cs = ax.tricontourf(*flatten_xs, flatten_y, cmap='viridis',
                                alpha=1)
            plt.colorbar(cs)

        else:
            ax.plot_trisurf(*flatten_xs, flatten_y, edgecolor='none',
                            cmap='viridis')
    elif plot_type == 'scatter':
        ax.scatter(*xs, Y, c='k', marker='.')
    else:
        raise ValueError(f'Unknown plot type choosen: {plot_type}')


def plot_data(X, Y, plot_type=None, **kwargs):
    """Plots data in 2d or 3d projection based on X shape."""
    ins = X.shape[1]
    if ins == 1:
        plot_2d_data(X, Y, plot_type)
    else:
        plot_3d_data(X, Y, plot_type, **kwargs)


def plot_fun(fun, *ranges, **kwargs):
    """Plots given function in given ranges."""
    ranges = [(hi, lo, min(1000, steps)) for hi, lo, steps in ranges]
    X, Y = generate_fun_samples(fun, *ranges)
    plot_data(X, Y, **kwargs)


def save_plot(fig, path, name):
    """Saves given figure."""
    fig.savefig(os.path.join(path, name))


def generate_fun_samples(fun, *ranges):
    """Generates function samples in given ranges."""
    xs = [np.linspace(lo, hi, steps) for lo, hi, steps in ranges]
    X = np.asarray([x for x in itertools.product(*xs)])
    Y = fun(X)
    return X, Y


def generate_random_fun_samples(fun, samples, *ranges, noise=False):
    """Generates random function samples in given ranges."""
    xs = [np.random.uniform(lo, hi, samples).T for lo, hi in ranges]
    X = np.column_stack(xs)
    Y = fun(X)
    mask = np.isfinite(Y).ravel()
    X = X[mask]
    Y = Y[mask]
    if noise:
        Y += np.random.random_sample(Y.shape) / 20
    return X, Y


def generate_fun_data(fun_def, dump=False, output_dir='', plot=False,
                      plot_show=False):
    """Generates input data for one function."""
    fun_name, fun, *ranges = fun_def
    Xs, Y = generate_fun_samples(fun, *ranges)
    name = f"{fun_name}"
    if dump:
        dump_fun_data(Xs, Y, output_dir, name + '.csv')
    if plot:
        fig = plt.figure()
        plot_fun(fun, *[(lo, hi, step * 10) for lo, hi, step in ranges])
        #plot_data(Xs, Y, 'scatter')
        save_plot(fig, output_dir, name + '.pdf')
        if plot_show:
            fig.show()
    return Xs, Y


def generate_data(functions):
    """Generates input data for experimenting with neural network libraries."""
    return [generate_fun_data(fun_def, True, 'output', True, True)
            for i, fun_def in enumerate(functions)]


R_FUN = ('damped_sine_wave', get_damped_sine_wave_fun(),
         (-4 * math.pi, 4 * math.pi, 100))

R2_FUN_1 = ('circular_vibration', get_circular_vibration_fun(),
            (-3 * math.pi, 3 * math.pi, 100),
            (-3 * math.pi, 3 * math.pi, 100))

R2_FUN_2 = ('rosenbrock', get_rosenbrock_fun(1, 100), (-2, 2, 100),
            (-1, 3, 100))

FUNS = [
    R_FUN,
    R2_FUN_1,
    R2_FUN_2
]


def run():
    # function definition: (x_size, y_size, function, (lo1, hi1, step1), ...)
    generate_data(FUNS)
    plt.show()


if __name__ == '__main__':
    run()
