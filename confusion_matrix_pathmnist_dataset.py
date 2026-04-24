import numpy as np
import matplotlib.pyplot as plt
from medmnist import INFO

cm_path = r"C:\bindhu\ML\outputs\confusion_matrix.csv"
output_path = r"C:\bindhu\ML\outputs\confusion_matrix.png"

cm = np.loadtxt(cm_path, delimiter=",", dtype=int)

info = INFO['pathmnist']
label_map = info['label']
class_names = [label_map[str(i)] for i in range(len(label_map))]

plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title("Confusion Matrix - PathMNIST")
plt.colorbar()

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45, ha="right")
plt.yticks(tick_marks, class_names)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print("Saved:", output_path)