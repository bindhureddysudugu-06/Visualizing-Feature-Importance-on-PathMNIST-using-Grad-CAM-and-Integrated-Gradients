import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import PathMNIST, INFO
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import numpy as np

# device usage 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# path, config
data_dir = r"C:\bindhu\ML\pathmnist_data"
model_path = r"C:\bindhu\ML\models\best_pathmnist_cnn.pth"
output_dir = r"C:\bindhu\ML\outputs\integrated_gradients"
correct_dir = os.path.join(output_dir, "correct")
incorrect_dir = os.path.join(output_dir, "incorrect")
os.makedirs(correct_dir, exist_ok=True)
os.makedirs(incorrect_dir, exist_ok=True)


# dataset information
info = INFO['pathmnist']
label_map = info['label']
class_names = [label_map[str(i)] for i in range(len(label_map))]
print("Class names:", class_names)


transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

# dataset and dataloaders
test_dataset = PathMNIST(
    split='test',
    root=data_dir,
    download=False,
    transform=transform,
    as_rgb=True
)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# model 
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

ig = IntegratedGradients(model)

#  graph plotting
def save_ig_plot(image_tensor, attr_map, true_label, pred_label, confidence, save_path):
    image = image_tensor.squeeze(0).cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = np.clip(image, 0, 1)

    plt.figure(figsize=(9, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(attr_map, cmap="jet")
    plt.title("Integrated Gradients")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(image)
    plt.imshow(attr_map, cmap="jet", alpha=0.5)
    plt.title("Overlay")
    plt.axis("off")

    true_name = class_names[true_label]
    pred_name = class_names[pred_label]
    plt.suptitle(f"True: {true_name} | Pred: {pred_name} | Conf: {confidence:.3f}", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# generate Integrated Gradients
num_correct_needed = 10
num_incorrect_needed = 10
correct_count = 0
incorrect_count = 0
sample_idx = 0

for images, labels in test_loader:
    if correct_count >= num_correct_needed and incorrect_count >= num_incorrect_needed:
        break

    images = images.to(device)
    labels = labels.squeeze().long().to(device)

    outputs = model(images)
    probs = F.softmax(outputs, dim=1)
    pred = torch.argmax(outputs, dim=1)
    confidence = probs[0, pred.item()].item()

    baseline = torch.zeros_like(images).to(device)

    attributions = ig.attribute(
        images,
        baselines=baseline,
        target=pred.item(),
        n_steps=50
    )

    # combine RGB channels into one attribution map
    attr = attributions.squeeze(0).detach().cpu().numpy()
    attr = np.transpose(attr, (1, 2, 0))
    attr = np.mean(np.abs(attr), axis=2)

    # normalize
    attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)

    true_label = labels.item()
    pred_label = pred.item()

    if pred_label == true_label and correct_count < num_correct_needed:
        save_path = os.path.join(
            correct_dir,
            f"sample_{sample_idx}_true_{class_names[true_label]}_pred_{class_names[pred_label]}.png"
        )
        save_ig_plot(images, attr, true_label, pred_label, confidence, save_path)
        correct_count += 1

    elif pred_label != true_label and incorrect_count < num_incorrect_needed:
        save_path = os.path.join(
            incorrect_dir,
            f"sample_{sample_idx}_true_{class_names[true_label]}_pred_{class_names[pred_label]}.png"
        )
        save_ig_plot(images, attr, true_label, pred_label, confidence, save_path)
        incorrect_count += 1

    sample_idx += 1

print("Integrated Gradients generation completed.")
print("Correct samples saved:", correct_count)
print("Incorrect samples saved:", incorrect_count)
print("Saved at:", output_dir)