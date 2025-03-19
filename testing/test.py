import torch
import torch.nn as nn
from models.model import DynamicNN
from data.data_loader import load_data

def inference_model(hparams, config):
    # ===== Device setup =====
    device = torch.device("cuda" if (torch.cuda.is_available() and config['device'] == 'cuda') else "cpu")
    print(f"Using device: {device}")

    # ===== Load Hyperparameters =====
    input_size = hparams["input_size"]
    hidden_layers = hparams["hidden_layers"]
    output_size = hparams["output_size"]
    test_size = hparams["test_size"]

    # ===== Model setup =====
    model = DynamicNN(input_size, hidden_layers, output_size).to(device)

    # ===== Load the model =====
    model.load_state_dict(torch.load("models/model.pth"))

    # ===== Load Test Data =====
    _, X_test, _, y_test = load_data(test_size)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    # ===== Evaluate the model =====
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to track gradients during evaluation
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)  # Get the predicted class
        correct = (predicted == y_test).sum().item()
        total = y_test.size(0)

    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    