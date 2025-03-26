import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from cnn import CNN
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

save_dir = "./images"
os.makedirs(save_dir, exist_ok=True)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
checkpoint = torch.load("saved_models/emnist_cnn_plus.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load test data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
test_set = datasets.EMNIST(root="./data", split="letters", train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# Evaluate model
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Accuracy: {100 * correct / total:.2f}%")

# Visualize the conv1's filter
conv1_weights = model.conv1.weight.data.cpu().numpy()                   # (out_channels, in_channels, height, width)
out_channels, in_channels, kernel_size, _ = conv1_weights.shape         # analysis the shape of the conv1_weights
num_cols = int(math.ceil(math.sqrt(out_channels)))
num_rows = int(math.ceil(out_channels / num_cols))
fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))

for out_idx in range(out_channels):         # traversal output channel
    row_idx = out_idx // num_cols
    col_idx = out_idx % num_cols

    if num_rows == 1 and num_cols == 1:
        ax = axes
    elif num_rows == 1:
        ax = axes[col_idx]
    elif num_cols == 1:
        ax = axes[row_idx]
    else:
        ax = axes[row_idx, col_idx]

    kernel = conv1_weights[out_idx, 0]                                      # get the kernel
    norm_kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())   # normalize the kernel
    ax.imshow(norm_kernel, cmap='gray', vmin=0, vmax=1)                     # display the kernel image
    ax.set_xticks([])
    ax.set_yticks([])
    # draw the border
    border = patches.Rectangle((-0.5, -0.5), kernel_size, kernel_size, 
                                linewidth=3, edgecolor='red', facecolor='none')
    ax.add_patch(border)
    # draw the grid
    for x in range(kernel_size + 1):
        ax.plot([x - 0.5, x - 0.5], [-0.5, kernel_size - 0.5], color='red', linewidth=2)
        ax.plot([-0.5, kernel_size - 0.5], [x - 0.5, x - 0.5], color='red', linewidth=2)
    ax.set_title(f"Filter {out_idx+1}")

# Delete empty axes
if out_channels < num_rows * num_cols:
    for i in range(out_channels, num_rows * num_cols):
        if num_rows == 1:
            fig.delaxes(axes[i])
        elif num_cols == 1:
            fig.delaxes(axes[i])
        else:
            fig.delaxes(axes[i // num_cols, i % num_cols])

plt.tight_layout()
save_path = os.path.join(save_dir, "plus_conv1_filter.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Conv1 filter visualization saved at {save_path}")
plt.show()