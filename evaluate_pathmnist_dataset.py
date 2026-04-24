import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import PathMNIST, INFO
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

# device usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# path information
data_dir = r"C:\bindhu\ML\pathmnist_data"
model_path = r"C:\bindhu\ML\models\best_pathmnist_cnn.pth"
output_dir = r"C:\bindhu\ML\outputs"
os.makedirs(output_dir, exist_ok=True)

# dataset information
info = INFO['pathmnist']
label_map = info['label']
class_names = [label_map[str(i)] for i in range(len(label_map))]
print("Class names:", class_names)

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

# dataset
test_dataset = PathMNIST(
    split='test',
    root=data_dir,
    download=False,
    transform=transform,
    as_rgb=True
)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# implementing the model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=9):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SimpleCNN(num_classes=9).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.squeeze().long().to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# accuracy metric
test_acc = accuracy_score(all_labels, all_preds)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

# report
report = classification_report(
    all_labels,
    all_preds,
    target_names=class_names,
    digits=4
)
print("\nClassification Report:\n")
print(report)

# saving the report
report_path = os.path.join(output_dir, "classification_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"Test Accuracy: {test_acc * 100:.2f}%\n\n")
    f.write(report)

# confusion matrix metrics
cm = confusion_matrix(all_labels, all_preds)

# saving the confusion matrix as csv
cm_csv_path = os.path.join(output_dir, "confusion_matrix.csv")
np.savetxt(cm_csv_path, cm, delimiter=",", fmt="%d")

print("\nSaved files:")
print("Classification report:", report_path)
print("Confusion matrix CSV:", cm_csv_path)