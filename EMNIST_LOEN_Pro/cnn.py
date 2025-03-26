import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryConv2d(nn.Conv2d):
    def forward(self, x):
        # Duplicate the weight tensor
        binary_weight = torch.sign(self.weight)
        return F.conv2d(x, binary_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Block 1
        self.conv1 = BinaryConv2d(1, 9, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(9)
        self.pool1 = nn.MaxPool2d(2, 2)
        # Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(9, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1, groups=64),
            nn.Conv2d(64, 64, 1, groups=16),
        )
        # SE block
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 16, 1),
            nn.SiLU(),
            nn.Conv2d(16, 64, 1),
            nn.Hardsigmoid()
        )
        # Spatial attention
        self.spatial_att = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, groups=64),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(64*6*6, 128),
            nn.Dropout(0.3),
            nn.SiLU(),
            nn.Linear(128, 27)
        )
        
    def forward(self, x):
        # Block 1
        x = F.silu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        # Block 2 with SE block
        x = self.conv2(x)
        se_weight = self.se(x)
        spatial_weight = self.spatial_att(x)
        x = x * se_weight * spatial_weight
        x = self.pool2(x)
        # Classifier
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Only run the following code when this file is run
if __name__ == "__main__":
    model = CNN()
    print(model)
    test_input = torch.randn(1, 1, 28, 28)  # Input image size: 28x28
    output = model(test_input)
    print(output.shape)                     # Output shape: [batch_size, 27] (27 classes)
