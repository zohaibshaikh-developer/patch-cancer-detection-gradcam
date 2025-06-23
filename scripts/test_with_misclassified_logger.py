import os
import torch
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm
from datetime import datetime
import csv

# Paths and model setup
val_dir = "dataset/binary_split/val"
model_path = "models/resnet18_epoch10_20250615_024251.pth"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False, num_classes=2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.7, 0.7, 0.7], [0.25, 0.25, 0.25])
])

# Dataset and loader
val_data = ImageFolder(val_dir, transform=transform)
val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

# Evaluation
correct = 0
total = 0
misclassified = []

for imgs, labels in tqdm(val_loader, desc="🔍 Evaluating"):
    imgs, labels = imgs.to(device), labels.to(device)
    outputs = model(imgs)
    preds = torch.argmax(outputs, dim=1)

    if preds == labels:
        correct += 1
    else:
        misclassified.append([val_data.imgs[total][0], preds.item(), labels.item()])
    total += 1

acc = correct / total * 100
print("✅ Accuracy: {:.2f}%".format(acc))
misclassified_path = os.path.join(log_dir, f"misclassified_{timestamp}.csv")

with open(misclassified_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "pred", "true"])
    writer.writerows(misclassified)

print(f"📁 Misclassified log saved to: {misclassified_path}")
