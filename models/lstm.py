import torch
import torch.nn as nn
from torch.utils.data import Dataset

# ===== LSTM Model =====
class lstm(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layers[0]
        self.lstm = nn.LSTM(input_size, hidden_layers[0], len(hidden_layers), batch_first=True)
        self.linear = nn.Linear(hidden_layers[0], output_size)
        self.train_dataloader = train_dataloader

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

# ===== Load Prediction Dataset =====
class train_dataloader(Dataset):
    def __init__(self, df, start_index, population, time_step, device):
        previous_overflow = max(start_index - time_step, 0)
        self.df = df.iloc[previous_overflow:start_index + population]
        self.time_step = time_step
        self.length = len(self.df) - self.time_step - 1
        self.device = device
        
    def __getitem__(self, index):
        previous_values = self.df.iloc[index:index + self.time_step].values
        previous_values = torch.tensor(previous_values)
        previous_values = previous_values.float().to(self.device)
        previous_values = previous_values.view(1, -1)
        target_values = self.df.iloc[index + self.time_step]
        target_values = torch.tensor(target_values).float().to(self.device)
        target_values = target_values
        return previous_values, target_values
    
    def __len__(self):
        return self.length