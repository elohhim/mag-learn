import torch
from torch import nn
from torch.optim import Adam

from experiment.base import Experiment

MAX_EPOCHS = 1000


class PyTorchExperiment(Experiment):
    tool_name = "pytorch"

    def __init__(self):
        super().__init__()

    def create_model(self, dataset):
        ins = dataset.x.shape[1]
        outs = dataset.y.shape[1]
        return nn.Sequential(nn.Linear(ins, self.NEURONS_BY_LAYER),
                             nn.ReLU(),
                             *[module
                               for _ in range(self.LAYERS_COUNT - 2)
                               for module in (nn.Linear(self.NEURONS_BY_LAYER,
                                                        self.NEURONS_BY_LAYER),
                                              nn.ReLU())
                               ],
                             nn.Linear(self.NEURONS_BY_LAYER, outs)
                             # No activation for outputs for regression problem
                             )

    def train(self, ctx):
        optimizer = Adam(ctx.model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        x_train = torch.from_numpy(ctx.x_scaled).float()
        y_train = torch.from_numpy(ctx.y_scaled).float()

        ctx.model.train()
        for epoch in range(1, MAX_EPOCHS + 1):
            optimizer.zero_grad()
            y_pred = ctx.model.forward(x_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print(f"Epoch #{epoch} Loss: {loss.item()}")
            if loss < ctx.epsilon:
                break
        ctx.model.eval()

    def simulate(self, ctx, x):
        x_torch = torch.from_numpy(x).float()
        return ctx.model.forward(x_torch).detach().numpy()


if __name__ == '__main__':
    PyTorchExperiment().run()
