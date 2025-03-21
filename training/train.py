import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from models.model import DynamicNN
from data.data_loader import load_data

def train_model(hparams, config):
    losses = []

    # ===== Device setup =====
    device = torch.device("cuda" if (torch.cuda.is_available() and config['device'] == 'cuda') else "cpu")
    print(f"Using device: {device}")

    # ===== Load Hyperparameters =====
    input_size = hparams["input_size"]
    hidden_layers = hparams["hidden_layers"]
    output_size = hparams["output_size"]
    learning_rate = hparams["learning_rate"]
    num_epochs = hparams["num_epochs"]
    test_size = hparams["test_size"]

    # ===== Model setup =====
    model = DynamicNN(input_size, hidden_layers, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ===== Load Data =====
    X_train, _, y_train, _ = load_data(test_size)
    X = X_train.to(device)
    y = y_train.to(device)

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
    
    # ===== Save the model =====
    torch.save(model.state_dict(), "models/model.pth")

    return losses