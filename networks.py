import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 64)
        self.fc7 = nn.Linear(64, 32)
        self.fc8 = nn.Linear(32, 16)
        self.fc9 = nn.Linear(16, 1)
        
    

        # define the activation function
        self.relu = nn.ReLU()
    

    def forward(self, t, S, q):
        # define the forward pass
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.tensor([t, S, q], device = device)
        x = self.fc1(x)
        x = self.relu(x)
        # add 5 more layers
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc7(x)
        x = self.relu(x)
        x = self.fc8(x)
        x = self.relu(x)
        x = self.fc9(x)

        return x
    