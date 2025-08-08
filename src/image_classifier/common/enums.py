from enum import Enum


class WeightInitiailizationMethod(Enum):
    XAIVER = "xaiver"
    HE = "he"


class ActiviationFunction(Enum):
    RELU = "relu"
    SIGMOId = "sigmoid"
    TANH = "tanh"
