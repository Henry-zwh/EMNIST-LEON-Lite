import torch
import torch.nn as nn
import torch.optim as optim
from load_emnist import train_loader, val_loader, test_loader
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
optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=5e-4)
scheduler = optim.lr_scheduler.OneCycleLR(  # OneCycleLR scheduler
    optimizer,
    max_lr=0.002,
    epochs=20,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,
    anneal_strategy='cos',
)

# model saving strategy
best_model_info = {
    'epoch': 0,
    'val_accuracy': 0.0,
    'model_state_dict': None,
    'optimizer_state_dict': None,
    'scheduler_state_dict': None
}
checkpoint_dir = "saved_models"
os.makedirs(checkpoint_dir, exist_ok=True)

# Model training
epochs = 15
total_train_time = 0.0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    epoch_start = time.time()

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        # model training
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        # conv1 weight binarization
        with torch.no_grad():
            model.conv1.weight.data = torch.sign(model.conv1.weight.data)
        # statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss  / len(train_loader)
    train_accuracy = correct  / total

    # Validation
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_running_loss / len(val_loader)
    val_accuracy = val_correct / val_total

    # Save best model
    if val_accuracy > best_model_info['val_accuracy']:
        best_model_info.update({
            'epoch': epoch+1,
            'val_accuracy': val_accuracy,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        })
        print(f"New best accuracy at epoch {epoch+1}: {val_accuracy:.4f}")

    # Print epoch results
    epoch_time = time.time() - epoch_start
    total_train_time += epoch_time
    print(f"Epoch {epoch+1}/{epochs} | "
          f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f} | "
          f"LR: {scheduler.get_last_lr()[0]:.6f} | "
          f"Time: {epoch_time:.2f}s")

# Save model
if best_model_info['model_state_dict'] is not None:
    torch.save(best_model_info, 
              os.path.join(checkpoint_dir, "emnist_cnn_beta.pth"))
    print(f"Saved best model from epoch {best_model_info['epoch']} with accuracy {best_model_info['val_accuracy']:.4f}")

print(f"\nTotal training time: {total_train_time:.2f} seconds")
print(f"Best validation accuracy: {best_model_info['val_accuracy']:.4f}")
print("Model saved successfully!")