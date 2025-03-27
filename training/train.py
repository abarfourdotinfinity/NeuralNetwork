import torch
import torch.nn as nn
import torch.optim as optim
from models import *
from data.data_loader import load_data
from training.training_loop import training_loop

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
    test_size = hparams["test_size"]

    # ===== Model setup =====
    model_class = model_dict[config["model"]]
    model = model_class(input_size, hidden_layers, output_size).to(device)

    criterion = loss_dict[config["loss"]]
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ===== Load Data =====
    A = load_data(test_size, config)

    # ===== Training loop =====
    losses = training_loop(model, A, optimizer, criterion, hparams, config, device)
    
    # ===== Save the model =====
    torch.save(model.state_dict(), f"saved_models/{config["model"]}.pth")

    return losses