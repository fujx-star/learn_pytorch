# 神经网络

import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


mynn = NeuralNetwork()
x = torch.tensor(1.0)
output = mynn(x)
print(output)
