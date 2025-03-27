import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryConv2d(nn.Conv2d):
    def forward(self, x):
        # Duplicate the weight tensor
        binary_weight = torch.sign(self.weight)
        return F.conv2d(x, binary_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class EnhancedAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(1, channels//8), 1),
            nn.SiLU(),
            nn.Conv2d(max(1, channels//8), channels, 1),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )
    def forward(self, x):
        # Channel attention
        ca = self.channel_att(x)
        # Spatial attention
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        sa = self.spatial_att(torch.cat([max_pool, avg_pool], dim=1))
        return x * ca * sa

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Block 1
        self.conv1 = BinaryConv2d(1, 9, kernel_size=3)
        self.pool1 = nn.MaxPool2d(3, 3)
        # Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(9, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU(),
        )
        # Attention
        self.attention  = EnhancedAttention(16)
        self.pool2 = nn.MaxPool2d(2, 2)
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 128),
            nn.Dropout(0.3),
            nn.SiLU(),
            nn.Linear(128, 27)
        )
        
    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.pool1(x)
        # Block 2 with attention
        x = self.conv2(x)
        att = self.attention(x)
        x = x * att
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
