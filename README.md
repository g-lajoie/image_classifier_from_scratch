# ImageNet Classifier from Scratch (ICS-Net)

This project is a **feedforward neural network** implemented **from scratch using only NumPy**, designed to perform **multiclass classification** on the [EMNIST dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset). It demonstrates how core neural network principles — such as forward propagation, backpropagation, gradient descent, and activation functions

## Features

- Built entirely using **NumPy**
- Supports **multiclass classification**
- Trained and tested on the **EMNIST dataset**
- Implements:
  - Custom weight initialization
  - ReLU and softmax activations
  - Categorical cross-entropy loss
  - Mini-batch stochastic gradient descent

## Requirements

- Python 3.13.5+
- NumPy

## Network Architecture

The architecture is fully configurable in code, by default it consist of

- Input Layer: 784 notes
- Hidden laeyers: [128, 64]
- Output Layer: 61 classes

Uses ReLU for hidden layers and output raw logits. Uses the softmax with categorical cross entropy loss function to evaluate the model.

## Author
Create by Ricky Lajoie.
