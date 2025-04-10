import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

transform = transforms.Compose([
    transforms.ToTensor(),                  # transform the image to tensor
    transforms.Normalize((0.5,), (0.5,))    # normalize the image (average: 0.5, std: 0.5)
])

# Download EMNIST dataset
emnist_train = datasets.EMNIST(root="./data", split="letters", train=True, download=True, transform=transform)
emnist_test = datasets.EMNIST(root="./data", split="letters", train=False, download=True, transform=transform)

# Split the training dataset into training and validation dataset
train_size = int(0.8 * len(emnist_train))
val_size = len(emnist_train) - train_size
train_dataset, val_dataset = random_split(emnist_train, [train_size, val_size])

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(emnist_test, batch_size=64, shuffle=False)

# Plot image and label
if __name__ == "__main__":
    image, label = emnist_train[0]              # Get the first image and label
    letter = chr(label + 96)                    # transform the label to letter
    plt.imshow(image.squeeze(), cmap="gray")    # squeeze the image tensor to remove the channel dimension
    plt.title(f"Label: {label} (Letter: {letter})")
    plt.axis("off")
    plt.show()

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of test samples: {len(emnist_test)}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")