from training.train import train_model
from utils import load_yaml, plot

if __name__ == "__main__":
    hparams = load_yaml("hparams.yaml")
    config = load_yaml("config.yaml")
    losses = train_model(hparams, config)
    plot(hparams["num_epochs"], losses)