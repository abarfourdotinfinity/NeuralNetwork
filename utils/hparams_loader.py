import yaml

def load_hparams(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
