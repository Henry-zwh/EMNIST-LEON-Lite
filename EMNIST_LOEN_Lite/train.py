import torch
import torch.nn as nn
import torch.optim as optim
from load_emnist import train_loader, test_loader
from cnn import CNN
import os
import time

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# CNN model initialization
model = CNN().to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)    # Learning rate scheduler

# Model training
epochs = 10
total_train_time = 0.0
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    epoch_start = time.time()

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.conv1.weight.data = torch.sign(model.conv1.weight.data)

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    scheduler.step()
    epoch_time = time.time() - epoch_start
    total_train_time += epoch_time
    print(f"Epoch {epoch+1}/{epochs} | "
          f"Loss: {running_loss/len(train_loader):.4f} | "
          f"Accuracy: {correct/total:.4f} | "
          f"LR: {scheduler.get_last_lr()[0]:.6f} | "
          f"Time: {epoch_time:.2f}s")

# Save model
os.makedirs("saved_models", exist_ok=True)
torch.save(model.state_dict(), "saved_models/emnist_cnn_lite.pth")
print(f"\nTotal training time: {total_train_time:.2f} seconds")
print("Model saved successfully!")
