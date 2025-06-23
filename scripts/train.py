import os
import csv
import time
import platform
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torch.cuda import amp

from load_data import get_dataloaders

# Setup
start_time = datetime.now()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

print("=" * 70)
print(f"🚀 Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"💻 Device: {device} - {gpu_name}")
print(f"🧠 System: {platform.system()} {platform.release()}")
print("=" * 70)

# Hyperparameters
EPOCHS = 10
BATCH_SIZE = 32
IMG_SIZE = 224
LR = 1e-4
data_path = "/scratch/zs00732/cancer_detection_camelyon17/dataset/binary_split"

# Dataloaders
train_loader, val_loader = get_dataloaders(data_path, batch_size=BATCH_SIZE, img_size=IMG_SIZE)

# Model, Loss, Optimizer
model = resnet18(pretrained=False, num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scaler = amp.GradScaler()

# Logging
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

metrics_path = f"logs/metrics_{start_time.strftime('%Y%m%d_%H%M%S')}.csv"
with open(metrics_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Train Loss", "Train Acc (%)", "Val Loss", "Val Acc (%)", "Duration (s)"])

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    epoch_start = time.time()

    print(f"\n📚 Epoch {epoch + 1}/{EPOCHS}")
    pbar = tqdm(train_loader, desc="🧪 Training", ncols=100)

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        with amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({
            "Loss": f"{running_loss / (total / BATCH_SIZE):.4f}",
            "Acc": f"{(correct / total) * 100:.2f}%"
        })

    train_loss = running_loss / len(train_loader)
    train_acc = (correct / total) * 100
    epoch_time = time.time() - epoch_start

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            with amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = (val_correct / val_total) * 100

    print(f"✅ Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Time: {epoch_time:.2f}s")

    with open(metrics_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, round(train_loss, 4), round(train_acc, 2),
                         round(val_loss, 4), round(val_acc, 2), round(epoch_time, 2)])

# Save model
end_time = datetime.now()
timestamp = end_time.strftime("%Y%m%d_%H%M%S")
model_path = f"models/resnet18_epoch{EPOCHS}_{timestamp}.pth"
torch.save(model.state_dict(), model_path)

# Training Log
log_path = f"logs/training_log_{timestamp}.txt"
with open(log_path, "w") as f:
    f.write("=" * 70 + "\n")
    f.write(f"🚀 Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"🛑 Training ended at:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"⏱️ Total time taken:    {end_time - start_time}\n")
    f.write(f"💻 Device: {device} - {gpu_name}\n")
    f.write(f"🧠 System: {platform.system()} {platform.release()}\n")
    f.write(f"📊 Final Validation Accuracy: {val_acc:.2f}%\n")
    f.write(f"📉 Final Validation Loss:     {val_loss:.4f}\n")
    f.write(f"💾 Model saved to: {model_path}\n")
    f.write("=" * 70 + "\n")

print(f"📝 Log saved to: {log_path}")
print(f"💾 Model saved to: {model_path}")



