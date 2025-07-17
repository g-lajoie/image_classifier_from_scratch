from enum import Enum


class WeightInitiailizationMethod(Enum):
    XAVIER = "xaiver"
    HE = "he"


class ActiviationFunction(Enum):
    RELU = "relu"
    SIGMOId = "sigmoid"
    TANH = "tanh"
