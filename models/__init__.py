import torch.nn as nn
import torch.optim as optim
from .neural_network import neural_network
from .lstm import lstm

model_dict = {
    "neural_network": neural_network,
    "lstm": lstm
}

loss_dict = {
    "cross_entropy": nn.CrossEntropyLoss(),
    "mse": nn.MSELoss(),
    "bce": nn.BCELoss(),
    # Add more if needed
}
