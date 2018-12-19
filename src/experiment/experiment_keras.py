from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.layers import Dense, Activation
from keras.models import Sequential

from experiment.base import Experiment

MAX_EPOCHS = 1000


class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='loss', value=0.01, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            print(f"Early stopping requires {self.monitor} available!")
            exit()

        if current < self.value:
            if self.verbose > 0:
                print(f"Epoch {epoch}: early stopping. Loss: {current}")
            self.model.stop_training = True


class KerasExperiment(Experiment):
    tool_name = "keras"

    def __init__(self):
        super().__init__()

    def create_model(self, dataset):
        input_shape = (dataset.x.shape[1],)
        out = dataset.y.shape[1]
        layers = [
            Dense(self.NEURONS_BY_LAYER, input_shape=input_shape),
            Activation('relu'),
            *[module
              for _ in range(self.LAYERS_COUNT - 2)
              for module in (Dense(self.NEURONS_BY_LAYER),
                             Activation('relu'))],
            Dense(out)
        ]
        return Sequential(layers=layers)

    def train(self, ctx):
        x_train = ctx.x_scaled
        y_train = ctx.y_scaled

        ctx.model.compile(optimizer='adam', loss='mse', metrics=['mse'])

        callbacks = [EarlyStoppingByLossVal(value=ctx.epsilon, verbose=1)]
        hist = ctx.model.fit(x_train, y_train, batch_size=100,
                             epochs=MAX_EPOCHS,
                             shuffle=False, verbose=0,
                             callbacks=callbacks)
        print(f"Final loss: {hist.history['loss'][-1]}")

    def simulate(self, ctx, x):
        return ctx.model.predict(x)


if __name__ == '__main__':
    KerasExperiment().run()
