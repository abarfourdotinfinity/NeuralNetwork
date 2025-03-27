from training.train import train_model
from testing.test import inference_model
from utils import load_yaml, plot

if __name__ == "__main__":
    hparams = load_yaml("hparams.yaml")
    config = load_yaml("config.yaml")

    # ===== Run training or testing =====
    if(config['run'] == 'train'):
        losses = train_model(hparams, config)
        plot(hparams["epochs"], losses)

    elif(config['run'] == 'test'):
        inference_model(hparams, config)
        
    else:
        print("Invalid run type. Please specify 'train' or 'test' in config.yaml")