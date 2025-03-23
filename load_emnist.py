import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),                  # transform the image to tensor
    transforms.Normalize((0.5,), (0.5,))    # normalize the image (average: 0.5, std: 0.5)
])

# Download EMNIST dataset
emnist_train = datasets.EMNIST(root="./data", split="letters", train=True, download=True, transform=transform)
emnist_test = datasets.EMNIST(root="./data", split="letters", train=False, download=True, transform=transform)

# Create DataLoader
from torch.utils.data import DataLoader
train_loader = DataLoader(emnist_train, batch_size=64, shuffle=True)
test_loader = DataLoader(emnist_test, batch_size=64, shuffle=False)

# Plot image and label
if __name__ == "__main__":
    image, label = emnist_train[0]              # Get the first image and label
    letter = chr(label + 96)                    # transform the label to letter
    plt.imshow(image.squeeze(), cmap="gray")    # squeeze the image tensor to remove the channel dimension
    plt.title(f"Label: {label} (Letter: {letter})")
    plt.axis("off")
    plt.show()
