import math
import pickle

import matplotlib.pyplot as plt
import numpy as np
import neurolab as nl
from sklearn import preprocessing

from src.experiment.data import get_damped_sine_wave_fun, plot_fun, \
    generate_fun_samples, plot_data, get_rosenbrock_fun, get_3d_sine_fun, \
    use_contour
from src.experiment.misc import timeit

LAYERS_COUNT = 4

NEURONS_BY_LAYER = 50

EPOCHS = 10000

PNG = '.png'

HIS = '.his'

NET = '.net'

plt.style.use('fast')


def run(from_file=False, to_file=False):
    functions = [(f'damped_sine_wave_{i}',
                  get_damped_sine_wave_fun(),
                  (0, 10*math.pi, 10 ** i + 1))
                 for i in range(1, 3)]
    functions.extend([
        ('rosenbrock_1', get_rosenbrock_fun(), (-2, 2, 11), (-1, 3, 11)),
       # ('rosenbrock_2', get_rosenbrock_fun(), (-2, 2, 21), (-1, 3, 21)),
        ('3d_sine_1', get_3d_sine_fun(), (0, 10 * math.pi, 11),
         (0, 10 * math.pi, 11)),
    ])
    for f_name, f, *ranges in functions:
        file_prefix = f'output/nl_{f_name}'
        print(f"### Using neurolab package for training neural network to "
              f"approximate {f_name.replace('_', ' ')} function on ranges: "
              f"{ranges}. Output files prefix: {file_prefix}")

        X, y = generate_fun_samples(f, *ranges)
        X_scaler = preprocessing.StandardScaler()
        X_scaled = X_scaler.fit_transform(X)
        y_scaler = preprocessing.MaxAbsScaler()
        y_scaled = y_scaler.fit_transform(y)

        if from_file:
            net = nl.load(file_prefix + NET)
            with open(file_prefix + HIS, 'rb') as file:
                training_history = pickle.load(file)
        else:
            layers = [NEURONS_BY_LAYER] * LAYERS_COUNT + [1]
            net = nl.net.newff([[-2, 2] for _ in ranges],
                               layers)

            @timeit
            def train():
                return nl.train.train_gdm(net, X_scaled, y_scaled,
                                          epochs=EPOCHS, show=100)

            training_history = train()
        if to_file:
            net.save(file_prefix + NET)
            with open(file_prefix + HIS, 'wb') as file:
                pickle.dump(training_history, file)

        def simulate(X_sim):
            return y_scaler.inverse_transform(net.sim(X_scaler.transform(X_sim)))

        plot_experiment_results(f, file_prefix, simulate, ranges, X, y,
                                training_history)


def plot_experiment_results(f, file_prefix, sim_fun, ranges, X, y,
                            training_history):
    """Plots experiment results."""
    fig = plt.figure()
    ax = fig.add_subplot(5, 1, 1)
    plot_training_history(training_history)
    projection = '3d' if len(ranges) == 2 and not use_contour else None
    ax = fig.add_subplot(5, 1, 2, projection=projection)
    plot_data(X, y, 'scatter', ax=ax)
    plot_data(X, sim_fun(X), 'scatter', ax=ax)
    ax = plt.subplot(5, 1, 3, projection=projection)
    dense_ranges = [(lo, hi, steps * 10) for lo, hi, steps in ranges]
    plot_fun(f, *dense_ranges, ax=ax)
    ax = plt.subplot(5, 1, 4, projection=projection)
    plot_fun(sim_fun, *dense_ranges, ax=ax)
    ax = plt.subplot(5, 1, 5, projection=projection)
    plot_fun(lambda X: (f(X) - sim_fun(X)), *dense_ranges, ax=ax)
    fig.savefig(file_prefix + PNG, dpi=600)


def plot_training_history(train_history):
    plt.plot(train_history)
    plt.xlabel('Epochs')
    plt.ylabel('Squared error')


if __name__ == '__main__':
    run(False, True)
    # run(True)
