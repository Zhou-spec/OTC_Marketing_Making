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
    def __init__(self, input_size, num_classes=10):
        super(ResNet, self).__init__()

        self.fc1 = nn.Linear(input_size, 64)
        self.layer1 = self.make_layer(64, 2)
        self.layer2 = self.make_layer(64, 2)
        self.layer3 = self.make_layer(64, 2)
        self.fc2 = nn.Linear(64, num_classes)
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

        return out



# this is the Resnet with convolution layers


