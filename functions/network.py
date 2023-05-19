import torch
import torch.nn as nn

# This is a plain MLP network
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
    

    

# this is the class for ResNet with only fully connected layers
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, activation=nn.ReLU()):
        super(ResidualBlock, self).__init__()

        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.activation = activation

    def forward(self, x):
        residual = x

        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)

        out += residual
        out = self.activation(out)

        return out


class ResNet(nn.Module):
    def __init__(self, input_size):
        super(ResNet, self).__init__()

        self.fc1 = nn.Linear(input_size, 256)
        self.layer1 = self.make_layer(256, 2)
        self.layer2 = self.make_layer(256, 2)
        self.layer3 = self.make_layer(256, 2)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.activation = nn.ReLU()

    def make_layer(self, out_features, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(out_features, out_features, nn.ReLU()))
        return nn.Sequential(*layers)

    def forward(self, t, S, q):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.tensor([t, S, q], device = device)
        out = self.fc1(x)
        out = self.activation(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.fc2(out)
        out = self.activation(out)
        out = self.fc3(out)
        out = self.activation(out)
        out = self.fc4(out)

        return out



# this is the Resnet with convolution layers
class ResidualBlock_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock_Conv, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        
        out += residual
        out = self.relu(out)
        
        return out

class ResNet_Conv(nn.Module):
    def __init__(self, input_channels, output_channels, num_blocks):
        super(ResNet_Conv, self).__init__()
        
        self.fc = nn.Linear(3, 128)
        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(ResidualBlock_Conv(output_channels, output_channels))
        
        self.conv2 = nn.Conv1d(output_channels, input_channels, kernel_size=3, stride=1, padding=1)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)
        
    def forward(self, t, S, q):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.tensor([t, S, q], device = device)        
        out = self.fc(x)
        out = out.unsqueeze(0)
        out = out.unsqueeze(0)
        out = self.conv1(out)
        out = self.relu(out)
        
        for block in self.blocks:
            out = block(out)
            
        out = self.conv2(out)
        out = self.relu(out)
        out = out.squeeze()
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)

        return out



# this is a simple CNN structure
class CNN(nn.Module):
    def __init__(self, input_size):
        super(CNN, self).__init__()
        self.fc = nn.Linear(input_size, 256)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels = 16, out_channels = 1, kernel_size = 3, stride = 1, padding=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        
    def forward(self, x):
        out = self.fc(x)
        out = out.unsqueeze(0)
        out = out.unsqueeze(0)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = out.squeeze()
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)

        return out