import math

import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split


from src.experiment.data import get_damped_sine_wave_fun, plot_fun, \
    generate_fun_samples, plot_data, get_rosenbrock_fun, get_3d_sine_fun, \
    use_contour
from src.experiment.misc import timeit

LAYERS_COUNT = 4

NEURONS_BY_LAYER = 50

EPOCHS = 100000

plt.style.use('ggplot')


def run():
    functions = [(f'damped_sine_wave_{i}',
                  get_damped_sine_wave_fun(),
                  (0, 10 * math.pi, 10 ** i + 1))
                 for i in range(1, 10)]
    functions.extend([
        ('rosenbrock_1', get_rosenbrock_fun(), (-2, 2, 11), (-1, 3, 11)),
        ('rosenbrock_2', get_rosenbrock_fun(), (-2, 2, 101), (-1, 3, 101)),
        ('3d_sine_1', get_3d_sine_fun(), (0, 10 * math.pi, 11),
         (0, 10 * math.pi, 11)),
        ('3d_sine_2', get_3d_sine_fun(), (0, 10 * math.pi, 101),
         (0, 10 * math.pi, 101)),

    ])
    for f_name, f, *ranges in functions:
        file_prefix = f'output/sklearn_{f_name}'
        print(f"### Using sklearn package for training neural network to "
              f"approximate {f_name.replace('_', ' ')} function on ranges: "
              f"{ranges}. Output files prefix: {file_prefix}")
        X, y = generate_fun_samples(f, *ranges)
        X_scaler = preprocessing.StandardScaler()
        X_scaled = X_scaler.fit_transform(X)
        y_scaler = preprocessing.MaxAbsScaler()
        y_scaled = y_scaler.fit_transform(y)

        mlp = MLPRegressor([NEURONS_BY_LAYER for _ in range(LAYERS_COUNT)],
                           max_iter=EPOCHS,
                           activation='relu',
                           learning_rate='adaptive')

        @timeit
        def train():
            mlp.fit(X_scaled, y_scaled.ravel())

        train()

        def simulate(X_predict):
            X_predict = X_scaler.transform(X_predict)
            y_predict = mlp.predict(X_predict).reshape(-1, 1)
            return y_scaler.inverse_transform(y_predict)

        plot_experiment_results(f, file_prefix, simulate, ranges, X, y)


def plot_experiment_results(f, file_prefix, f_sim, ranges, X, y):
    """Plots experiment results."""
    fig = plt.figure()
    #ax = fig.add_subplot(5, 1, 1)
    #plot_training_history(training_history)
    projection = '3d' if len(ranges) == 2 and not use_contour else None
    #ax = fig.add_subplot(4, 1, 1, projection=projection)
    #plot_data(X, y, 'scatter', ax=ax)
    #plot_data(X, f_sim(X), 'scatter', ax=ax)
    ax = plt.subplot(3, 1, 1, projection=projection)
    plot_ranges = [(lo, hi, 1000) for lo, hi, steps in ranges]
    plot_fun(f, *plot_ranges, ax=ax)
    ax = plt.subplot(3, 1, 2, projection=projection)
    plot_fun(f_sim, *plot_ranges, ax=ax)
    ax = plt.subplot(3, 1, 3, projection=projection)
    plot_fun(lambda X: (f(X) - f_sim(X)), *plot_ranges, ax=ax)
    fig.savefig(f'{file_prefix}.png', dpi=600)


def scale_samples(samples):
    """Scale samples to (-1, 1) range."""
    scaler = preprocessing.MaxAbsScaler()
    scaled_samples = scaler.fit_transform(samples)
    return scaled_samples, scaler


def plot_training_history(train_history):
    plt.plot(train_history)
    plt.xlabel('Epochs')
    plt.ylabel('Squared error')


if __name__ == '__main__':
    run()
