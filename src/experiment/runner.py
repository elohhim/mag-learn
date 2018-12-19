#!/usr/bin/env python
from experiment.experiment_keras import KerasExperiment
from experiment.experiment_pytorch import PyTorchExperiment
from experiment.experiment_pytorch_gpu import PyTorchGPUExperiment
from experiment.experiment_sklearn import SklearnExperiment

if __name__ == '__main__':
    experiments = [
        KerasExperiment(),
        SklearnExperiment(),
        PyTorchExperiment(),
        PyTorchExperiment(batch_size=100),
        PyTorchGPUExperiment()
    ]
    for experiment in experiments:
        experiment.run()
