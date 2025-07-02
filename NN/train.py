import torch
import torch.nn as nn

class HexaPawnNet(nn.Module):
    
    def __init__(self, input_size=9, hidden_size=64, output_size=9):
        super(HexaPawnNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)
    
