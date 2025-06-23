import os
import csv
import torch
import platform
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader

# Paths
data_path = "/scratch/zs00732/cancer_detection_camelyon17/dataset/binary_split/val"
model_path = "models/resnet18_epoch10_20250615_024251.pth"  # 🔁 Update to your latest .pth
output_log_folder = "logs"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(output_log_folder, f"test_log_{timestamp}.txt")
csv_file = os.path.join(output_log_folder, f"test_metrics_{timestamp}.csv")

os.makedirs(output_log_folder, exist_ok=True)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

print("=" * 70)
print(f"🧪 Test started at: {timestamp}")
print(f"💻 Device: {device} - {gpu_name}")
print(f"🧠 System: {platform.system()} {platform.release()}")
print("=" * 70)

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.7, 0.7, 0.7], [0.25, 0.25, 0.25])
])

# Data
test_dataset = datasets.ImageFolder(root=data_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Model
model = models.resnet18(pretrained=False, num_classes=2).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
criterion = nn.CrossEntropyLoss()

# Eval
correct, total, total_loss = 0, 0, 0.0
class_correct = [0] * 2
class_total = [0] * 2

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="🔍 Evaluating", ncols=100):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for i in range(len(labels)):
            label = labels[i]
            class_total[label] += 1
            if predicted[i] == label:
                class_correct[label] += 1

# Final metrics
acc = correct / total * 100
avg_loss = total_loss / len(test_loader)
per_class_acc = [class_correct[i] / class_total[i] * 100 if class_total[i] else 0.0 for i in range(2)]

# Save logs
with open(log_file, "w") as f:
    f.write("=" * 70 + "\n")
    f.write(f"🧪 Test Time: {timestamp}\n")
    f.write(f"💻 Device: {device} - {gpu_name}\n")
    f.write(f"📉 Test Loss: {avg_loss:.4f}\n")
    f.write(f"📊 Test Accuracy: {acc:.2f}%\n")
    f.write(f"🎯 Per-Class Accuracy: Normal={per_class_acc[0]:.2f}%, Anomaly={per_class_acc[1]:.2f}%\n")
    f.write("=" * 70 + "\n")

with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Total Samples", "Correct", "Accuracy (%)", "Loss"])
    writer.writerow([total, correct, round(acc, 2), round(avg_loss, 4)])
    writer.writerow(["Per-Class Accuracy", "Normal", round(per_class_acc[0], 2)])
    writer.writerow(["", "Anomaly", round(per_class_acc[1], 2)])

print(f"📝 Test log saved to: {log_file}")
print(f"📊 Test metrics saved to: {csv_file}")
print("=" * 70)
print(f"✅ Test Accuracy: {acc:.2f}% | Loss: {avg_loss:.4f}")
print(f"🎯 Per-Class Accuracy: Normal={per_class_acc[0]:.2f}%, Anomaly={per_class_acc[1]:.2f}%")



