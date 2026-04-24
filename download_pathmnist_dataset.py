import os
from medmnist import PathMNIST

#saving the downloaded dataset at the given path
save_dir = r"C:\bindhu\ML\pathmnist_data"
os.makedirs(save_dir, exist_ok=True)

#splitting the dataset into trainig, validation and testing data
train_dataset = PathMNIST(split='train', download=True, root=save_dir)
val_dataset = PathMNIST(split='val', download=True, root=save_dir)
test_dataset = PathMNIST(split='test', download=True, root=save_dir)

print("Download completed successfully.")
print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Test samples:", len(test_dataset))
print("Saved at:", save_dir)