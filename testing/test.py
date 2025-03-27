import torch
from models import *
from data.data_loader import load_data
from testing.testing_model import test_model

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
    model_class = model_dict[config["model"]]
    model = model_class(input_size, hidden_layers, output_size).to(device)

    # ===== Load the model =====
    model.load_state_dict(torch.load(f"saved_models/{config["model"]}.pth"))

    # ===== Load Test Data =====
    A = load_data(test_size, config)

    # ===== Evaluate the model =====
    accuracy = test_model(A, device, model, hparams, config)

    
    print(f"Test Accuracy: {accuracy:.2f}%")
    