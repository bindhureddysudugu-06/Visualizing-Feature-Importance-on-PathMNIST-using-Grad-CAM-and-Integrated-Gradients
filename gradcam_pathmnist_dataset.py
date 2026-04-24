import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import PathMNIST, INFO
import matplotlib.pyplot as plt
import numpy as np

# device usage 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# paths information
data_dir = r"C:\bindhu\ML\pathmnist_data"
model_path = r"C:\bindhu\ML\models\best_pathmnist_cnn.pth"
output_dir = r"C:\bindhu\ML\outputs\gradcam"
correct_dir = os.path.join(output_dir, "correct")
incorrect_dir = os.path.join(output_dir, "incorrect")
os.makedirs(correct_dir, exist_ok=True)
os.makedirs(incorrect_dir, exist_ok=True)

# dataset information 
info = INFO['pathmnist']
label_map = info['label']
class_names = [label_map[str(i)] for i in range(len(label_map))]
print("Class names:", class_names)

# transforms
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

# Grad-CAM hook variables

activations = None
gradients = None

def forward_hook(module, input, output):
    global activations
    activations = output

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

# Hook the last convolution layer = features[3]
target_layer = model.features[3]
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)


#  graph plotting
def save_gradcam_plot(image_tensor, cam, true_label, pred_label, confidence, save_path):
    image = image_tensor.squeeze(0).cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = np.clip(image, 0, 1)

    plt.figure(figsize=(9, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(cam, cmap="jet")
    plt.title("Grad-CAM")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(image)
    plt.imshow(cam, cmap="jet", alpha=0.5)
    plt.title("Overlay")
    plt.axis("off")

    true_name = class_names[true_label]
    pred_name = class_names[pred_label]
    plt.suptitle(f"True: {true_name} | Pred: {pred_name} | Conf: {confidence:.3f}", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# generate Grad-CAM

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

    model.zero_grad()
    outputs = model(images)

    probs = F.softmax(outputs, dim=1)
    pred = torch.argmax(outputs, dim=1)
    confidence = probs[0, pred.item()].item()

    score = outputs[0, pred.item()]
    score.backward()

    # activations: [1, C, H, W]
    # gradients:   [1, C, H, W]
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])   # [C]
    feature_maps = activations.squeeze(0).detach().clone()     # [C, H, W]

    for i in range(feature_maps.shape[0]):
        feature_maps[i, :, :] *= pooled_gradients[i]

    cam = torch.mean(feature_maps, dim=0)
    cam = F.relu(cam)

    cam = cam.cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    cam_tensor = torch.tensor(cam).unsqueeze(0).unsqueeze(0)
    cam_tensor = F.interpolate(cam_tensor, size=(28, 28), mode='bilinear', align_corners=False)
    cam = cam_tensor.squeeze().numpy()

    true_label = labels.item()
    pred_label = pred.item()

    if pred_label == true_label and correct_count < num_correct_needed:
        save_path = os.path.join(
            correct_dir,
            f"sample_{sample_idx}_true_{class_names[true_label]}_pred_{class_names[pred_label]}.png"
        )
        save_gradcam_plot(images, cam, true_label, pred_label, confidence, save_path)
        correct_count += 1

    elif pred_label != true_label and incorrect_count < num_incorrect_needed:
        save_path = os.path.join(
            incorrect_dir,
            f"sample_{sample_idx}_true_{class_names[true_label]}_pred_{class_names[pred_label]}.png"
        )
        save_gradcam_plot(images, cam, true_label, pred_label, confidence, save_path)
        incorrect_count += 1

    sample_idx += 1

print("Grad-CAM generation completed.")
print("Correct samples saved:", correct_count)
print("Incorrect samples saved:", incorrect_count)
print("Saved at:", output_dir)