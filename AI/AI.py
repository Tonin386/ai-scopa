import torch
import torch.nn as nn

if torch.cuda.is_available():
    torch.set_default_device("cuda:0")
    torch.cuda.set_device("cuda:0")
    device = "cuda:0"

class AI(nn.Module):
    def __init__(self, state_size, action_size):
        super(AI, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, action_size)
        ).to(device)

    def forward(self, x):
        return self.network(x)