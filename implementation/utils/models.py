import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size: int = 28*28*1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3))
        self.conv2 = nn.Conv2d(32, 64, (3, 3))
        self.pool1 = nn.MaxPool2d((2, 2))
        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout(x)

        x = x.view(-1, 12 * 12 * 64)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class AMCNN(nn.Module):
    def __init__(self):
        super(AMCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))

        x = x.view(-1, 320)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class CNNBase(nn.Module):
    def __init__(self):
        super(CNNBase, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(2)

    def forward(self, x):
        x, first_indices = F.max_pool2d(self.conv1(x), 2, return_indices=True)
        x = F.relu(x)

        x, second_indices = F.max_pool2d(self.conv2(x), 2, return_indices=True)
        x = F.relu(x)

        return x, first_indices, second_indices


class SeparableCNN(nn.Module):
    def __init__(self):
        super(SeparableCNN, self).__init__()
        self.base = CNNBase()
        self.classifier = CNNClassifier()

    def forward(self, x):
        x, first, second = self.base(x)
        x = x.view(-1, 320)
        x = self.classifier(x)

        return x, first, second


class DeConvNet(nn.Module):
    def __init__(self):
        super(DeConvNet, self).__init__()
        self.unpool1 = nn.MaxUnpool2d(2)
        self.deconv1 = nn.ConvTranspose2d(20, 10, 5)
        self.unpool2 = nn.MaxUnpool2d(2)
        self.deconv2 = nn.ConvTranspose2d(10, 1, 5)

    def forward(self, x, first_indices, second_indices):
        x = self.deconv1(self.unpool1(F.relu(x), first_indices))
        x = self.deconv2(self.unpool2(F.relu(x), second_indices))

        return x
