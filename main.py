from training.train import train_model
from utils import load_hparams, plot

if __name__ == "__main__":
    hparams = load_hparams("hparams.yaml")
    losses = train_model(hparams)
    plot(hparams["num_epochs"], losses)
