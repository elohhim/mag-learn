import datetime
import math
import pathlib
from abc import ABC, abstractmethod
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

from experiment.data import use_contour, plot_fun, get_damped_sine_wave_fun, \
    generate_random_fun_samples, plot_data, get_rosenbrock_fun, \
    get_circular_vibration_fun
from experiment.misc import timeit

ExperimentDataset = namedtuple("ExperimentDataset",
                               "fun_name fun samples ranges x y")

ExperimentContext = namedtuple("ExperimentContext",
                               ("dataset x_scaler y_scaler x_scaled y_scaled "
                                "model epsilon"))

ExperimentResult = namedtuple("ExperimentResult", "time history")


class Experiment(ABC):
    LAYERS_COUNT = 4

    NEURONS_BY_LAYER = 50

    def __init__(self):
        super().__init__()

    def run(self):
        """Run experiment of training NN to approximate some mathematical
        functions."""
        datasets = self.generate_datasets(False)
        output_dir = self.create_output_dir()
        for dataset in datasets:
            file_prefix = f"{self.tool_name}_{dataset.fun_name.replace(' ', '_')}_{dataset.samples}"
            print(f"### Using {self.tool_name} for training neural network to "
                  f"approximate {dataset.fun_name} function. Output files: "
                  f"{file_prefix}...")
            x_scaler, y_scaler = (preprocessing.StandardScaler(),
                                  preprocessing.StandardScaler())
            x_scaled, y_scaled = (x_scaler.fit_transform(dataset.x),
                                  y_scaler.fit_transform(dataset.y))
            model = self.create_model(dataset)
            ctx = ExperimentContext(dataset, x_scaler, y_scaler, x_scaled,
                                    y_scaled, model, 1e-4)
            self._train(ctx)
            self.plot_results(ctx, pathlib.Path(output_dir, file_prefix))

    @timeit
    def _train(self, ctx):
        self.train(ctx)

    def _simulate(self, ctx, x):
        x_scaled = ctx.x_scaler.transform(x)
        y_scaled = self.simulate(ctx, x_scaled)
        return ctx.y_scaler.inverse_transform(y_scaled)

    @abstractmethod
    def create_model(self, dataset: ExperimentDataset):
        ...

    @abstractmethod
    def train(self, ctx: ExperimentContext):
        ...

    @abstractmethod
    def simulate(self, ctx: ExperimentContext, x: np.array):
        ...

    @property
    @abstractmethod
    def tool_name(self):
        ...

    def generate_datasets(self, noise=False):
        fun_definitions = []
        fun_definitions.extend(
            (f'damped sine wave', get_damped_sine_wave_fun(), n,
             (-4 * math.pi, 4 * math.pi))
            for n in [100, 10_000, 1_000_000]
        )
        fun_definitions.extend(
            ('rosenbrock', get_rosenbrock_fun(), n, (-2, 2), (-1, 3))
            for n in [100, 10_000, 1_000_000])
        fun_definitions.extend(
            ('circular_vibration', get_circular_vibration_fun(), n,
             (-3 * math.pi, 3 * math.pi), (-3 * math.pi, 3 * math.pi))
            for n in [100, 10_000, 1_000_000])
        datasets = []
        for fun_name, fun, samples, *ranges in fun_definitions:
            x, y = generate_random_fun_samples(fun, samples, *ranges,
                                               noise=noise)
            datasets.append(
                ExperimentDataset(fun_name, fun, samples, ranges, x, y))
        return datasets

    def plot_results(self, ctx: ExperimentContext, output_path):
        """Plots experiment results."""
        dataset = ctx.dataset
        fig = plt.figure()
        projection = '3d' if dataset.x.shape[
                                 1] == 2 and not use_contour else None
        ax = plt.subplot(3, 1, 1, projection=projection)
        plot_ranges = [(lo, hi, 100) for lo, hi in dataset.ranges]
        plot_fun(dataset.fun, *plot_ranges, ax=ax)
        ax = plt.subplot(3, 1, 2, projection=projection)
        plot_fun(lambda x: self._simulate(ctx, x), *plot_ranges, ax=ax)
        ax = plt.subplot(3, 1, 3, projection=projection)
        plot_fun(lambda x: (dataset.fun(x) - self._simulate(ctx, x)),
                 *plot_ranges, ax=ax)
        fig.savefig(f'{output_path}.png', dpi=600)

    @staticmethod
    def create_output_dir():
        """Creates output directory."""
        date = datetime.datetime.now().isoformat()
        path = pathlib.Path("output", date.replace(':', '_').replace('.', '_'))
        path.mkdir(parents=True, exist_ok=False)
        return path
