import numpy as np
import torch
import torch.nn as nn


def activation_maximization(model, class_num, epochs):
    criterion = nn.CrossEntropyLoss()
    x = torch.rand((1, 28, 28))

    for i in range(epochs):
        output = model(x)

        loss = criterion(output, np.array([class_num]))
        loss.backward()
