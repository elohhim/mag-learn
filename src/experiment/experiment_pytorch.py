import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from experiment.base import Experiment

MAX_EPOCHS = 1000


class PyTorchExperiment(Experiment):
    tool_name = "pytorch"

    def __init__(self, batch_size='auto', verbose=0):
        super().__init__()
        self._batch_size = batch_size
        self._verbose = verbose

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
        batch_size = x_train.shape[0] if self._batch_size == 'auto' else self._batch_size
        dataset = TensorDataset(x_train, y_train)
        dataloader = DataLoader(dataset, batch_size)

        ctx.model.train()
        loss_history = []
        for epoch in range(1, MAX_EPOCHS + 1):
            loss = 1
            for x_batch, y_batch in dataloader:
                optimizer.zero_grad()
                y_pred = ctx.model.forward(x_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
            loss_history.append(loss.item())
            if self._verbose and epoch % 10 == 0:
                print(f"Epoch #{epoch} Loss: {loss.item()}")
            if loss < ctx.epsilon:
                f"Epoch {epoch}: early stopping. Loss: {loss.item()}"
                break
        ctx.model.eval()
        return loss_history

    def simulate(self, ctx, x):
        x_torch = torch.from_numpy(x).float()
        return ctx.model.forward(x_torch).detach().numpy()


if __name__ == '__main__':
    PyTorchExperiment(batch_size=100).run()
