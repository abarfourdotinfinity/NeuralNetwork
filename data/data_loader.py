import torch

def get_dummy_data(input_size, output_size, num_samples=100):
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, output_size, (num_samples,))
    return X, y