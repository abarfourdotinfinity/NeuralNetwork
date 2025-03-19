import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from models.model import DynamicNN
from data.data_loader import get_dummy_data

def train_model(hparams):
    losses = []

    # ===== Load Hyperparameters =====
    input_size = hparams["input_size"]
    hidden_size = hparams["hidden_size"]
    num_hidden_layers = hparams["num_hidden_layers"]
    output_size = hparams["output_size"]
    learning_rate = hparams["learning_rate"]
    num_epochs = hparams["num_epochs"]

    # ===== Model setup =====
    model = DynamicNN(input_size, hidden_size, num_hidden_layers, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ===== Load Data =====
    X, y = get_dummy_data(input_size, output_size)

    # ===== Training loop =====
    for epoch in range(num_epochs):
        outputs = model(X)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    return losses