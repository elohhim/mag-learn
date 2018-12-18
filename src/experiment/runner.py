#!/usr/bin/env python
from experiment.example_pytorch import PyTorchExperiment
from experiment.example_pytorch_gpu import PyTorchGPUExperiment
from experiment.example_sklearn import SklearnExperiment

if __name__ == '__main__':
    PyTorchExperiment().run()
    PyTorchGPUExperiment().run()
    SklearnExperiment().run()
