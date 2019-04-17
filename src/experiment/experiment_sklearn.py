import numpy as np
from sklearn.neural_network import MLPRegressor

from experiment.base import Experiment, ExperimentContext, ExperimentDataset

MAX_EPOCHS = 1000


class SklearnExperiment(Experiment):

    def create_model(self, dataset: ExperimentDataset):
        return MLPRegressor([self.NEURONS_BY_LAYER
                             for _ in range(self.LAYERS_COUNT)],
                            max_iter=MAX_EPOCHS,
                            activation='relu',
                            solver='adam',
                            learning_rate='adaptive',)

    def train(self, ctx: ExperimentContext):
        ctx.model.fit(ctx.x_scaled, ctx.y_scaled.ravel(), )
        return ctx.model.loss_curve_

    def simulate(self, ctx: ExperimentContext, x: np.array):
        return ctx.model.predict(x).reshape(-1, 1)

    @property
    def tool_name(self):
        return "scikit_learn"


if __name__ == '__main__':
    SklearnExperiment().run()
