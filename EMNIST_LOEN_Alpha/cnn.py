import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryConv2d(nn.Conv2d):
    def forward(self, x):
        # Duplicate the weight tensor
        binary_weight = (self.weight > 0).float()
        return F.conv2d(x, binary_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = BinaryConv2d(1, 1, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(1)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(8, 4, 1),
            nn.ReLU(),
            nn.Conv2d(4, 8, 1),
            nn.Hardsigmoid()
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc = nn.Sequential(
            nn.Linear(8*6*6, 64),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(64, 27)
        )
        
    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        # Block 2 with SE block
        x = F.relu(self.bn2(self.conv2(x)))
        se_weight = self.se(x)
        x = x * se_weight
        x = self.pool2(x)
        # Classifier
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Only run the following code when this file is run
if __name__ == "__main__":
    model = CNN()
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e3:.1f}K")
    test_input = torch.randn(1, 1, 28, 28)  # Input image size: 28x28
    output = model(test_input)
    print(output.shape)                     # Output shape: [batch_size, 27] (27 classes)
