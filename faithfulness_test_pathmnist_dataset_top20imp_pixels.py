import os
import csv
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import PathMNIST, INFO
from captum.attr import IntegratedGradients
import numpy as np

# device usage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# paths information
data_dir = r"C:\bindhu\ML\pathmnist_data"
model_path = r"C:\bindhu\ML\models\best_pathmnist_cnn.pth"
output_dir = r"C:\bindhu\ML\outputs\faithfulness"
os.makedirs(output_dir, exist_ok=True)

csv_path = os.path.join(output_dir, "faithfulness_results.csv")

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

# Grad-CAM hook variables

activations = None
gradients = None

def forward_hook(module, input, output):
    global activations
    activations = output

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

target_layer = model.features[3]
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# Integrated gradient object

ig = IntegratedGradients(model)


# helping functions
def get_prediction_and_confidence(input_tensor):
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0, pred].item()
    return pred, conf

def compute_gradcam_map(input_tensor, target_class):
    global activations, gradients
    model.zero_grad()

    output = model(input_tensor)
    score = output[0, target_class]
    score.backward(retain_graph=True)

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])   # [C]
    feature_maps = activations.squeeze(0).detach().clone()     # [C,H,W]

    for i in range(feature_maps.shape[0]):
        feature_maps[i] *= pooled_gradients[i]

    cam = torch.mean(feature_maps, dim=0)
    cam = F.relu(cam)
    cam = cam.detach().cpu().numpy()

    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    cam_tensor = torch.tensor(cam, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    cam_tensor = F.interpolate(cam_tensor, size=(28, 28), mode='bilinear', align_corners=False)
    cam = cam_tensor.squeeze().numpy()

    return cam

def compute_ig_map(input_tensor, target_class):
    baseline = torch.zeros_like(input_tensor).to(device)

    attributions = ig.attribute(
        input_tensor,
        baselines=baseline,
        target=target_class,
        n_steps=50
    )

    attr = attributions.squeeze(0).detach().cpu().numpy()   # [C,H,W]
    attr = np.transpose(attr, (1, 2, 0))                    # [H,W,C]
    attr = np.mean(np.abs(attr), axis=2)                    # [H,W]

    attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
    return attr

def create_top_mask(score_map, top_fraction=0.2):
    """
    score_map: [H,W]
    returns binary mask where top important pixels are 1
    """
    flat = score_map.flatten()
    k = max(1, int(len(flat) * top_fraction))
    threshold = np.partition(flat, -k)[-k]
    mask = (score_map >= threshold).astype(np.float32)
    return mask

def create_random_mask(shape, top_fraction=0.2):
    h, w = shape
    total = h * w
    k = max(1, int(total * top_fraction))

    flat_mask = np.zeros(total, dtype=np.float32)
    idx = np.random.choice(total, size=k, replace=False)
    flat_mask[idx] = 1.0
    return flat_mask.reshape(h, w)

def apply_mask_to_image(input_tensor, binary_mask):
    """
    input_tensor: [1,3,H,W]
    binary_mask: [H,W] with 1 = mask/remove important region
    """
    mask_tensor = torch.tensor(binary_mask, dtype=torch.float32, device=input_tensor.device)
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    masked_image = input_tensor.clone() * (1.0 - mask_tensor)
    return masked_image

# experiment settings
num_samples_to_test = 20
top_fraction = 0.2   # top 20% important pixels
sample_idx = 0

results = []

gradcam_drops = []
ig_drops = []
random_drops = []

print(f"Running faithfulness test on {num_samples_to_test} test samples...")
print(f"Masking top {top_fraction * 100:.0f}% pixels")

for images, labels in test_loader:
    if sample_idx >= num_samples_to_test:
        break

    images = images.to(device)
    labels = labels.squeeze().long().to(device)

    pred_class, original_conf = get_prediction_and_confidence(images)

    # explanation maps for predicted class
    gradcam_map = compute_gradcam_map(images, pred_class)
    ig_map = compute_ig_map(images, pred_class)
    rand_map = create_random_mask(gradcam_map.shape, top_fraction=top_fraction)

    # creating masks
    gradcam_mask = create_top_mask(gradcam_map, top_fraction=top_fraction)
    ig_mask = create_top_mask(ig_map, top_fraction=top_fraction)

    # applying masks
    gradcam_masked_img = apply_mask_to_image(images, gradcam_mask)
    ig_masked_img = apply_mask_to_image(images, ig_mask)
    random_masked_img = apply_mask_to_image(images, rand_map)

    # evaluate masked confidence for same original predicted class
    with torch.no_grad():
        gradcam_outputs = model(gradcam_masked_img)
        ig_outputs = model(ig_masked_img)
        random_outputs = model(random_masked_img)

        gradcam_probs = F.softmax(gradcam_outputs, dim=1)
        ig_probs = F.softmax(ig_outputs, dim=1)
        random_probs = F.softmax(random_outputs, dim=1)

        gradcam_conf = gradcam_probs[0, pred_class].item()
        ig_conf = ig_probs[0, pred_class].item()
        random_conf = random_probs[0, pred_class].item()

    gradcam_drop = original_conf - gradcam_conf
    ig_drop = original_conf - ig_conf
    random_drop = original_conf - random_conf

    gradcam_drops.append(gradcam_drop)
    ig_drops.append(ig_drop)
    random_drops.append(random_drop)

    results.append({
        "sample_index": sample_idx,
        "true_label": class_names[labels.item()],
        "predicted_label": class_names[pred_class],
        "original_confidence": original_conf,
        "gradcam_masked_confidence": gradcam_conf,
        "ig_masked_confidence": ig_conf,
        "random_masked_confidence": random_conf,
        "gradcam_confidence_drop": gradcam_drop,
        "ig_confidence_drop": ig_drop,
        "random_confidence_drop": random_drop
    })

    print(
        f"Sample {sample_idx:02d} | "
        f"True: {class_names[labels.item()]} | Pred: {class_names[pred_class]} | "
        f"Orig: {original_conf:.4f} | "
        f"GC Drop: {gradcam_drop:.4f} | "
        f"IG Drop: {ig_drop:.4f} | "
        f"Rand Drop: {random_drop:.4f}"
    )

    sample_idx += 1

# save CSV

with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

# summary
avg_gradcam_drop = float(np.mean(gradcam_drops))
avg_ig_drop = float(np.mean(ig_drops))
avg_random_drop = float(np.mean(random_drops))

print("\nFaithfulness test completed.")
print("Results saved to:", csv_path)
print(f"Average Grad-CAM confidence drop: {avg_gradcam_drop:.4f}")
print(f"Average Integrated Gradients confidence drop: {avg_ig_drop:.4f}")
print(f"Average Random Masking confidence drop: {avg_random_drop:.4f}")