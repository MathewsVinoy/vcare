import os
import pandas as pd
import torch
import torch.optim as optim
from torch import nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

class SkinCancerDataset(Dataset):
    def __init__(self, csv_file, img_dirs, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dirs = img_dirs if isinstance(img_dirs, list) else [img_dirs]
        self.transform = transform

        # Map the 7 labels to binary (0 = no_cancer, 1 = cancer)
        self.label_map = {
            "mel": 1,   # Melanoma
            "bcc": 1,   # Basal Cell Carcinoma
            "akiec": 1, # Bowen’s / SCC in situ
            "nv": 0,
            "bkl": 0,
            "df": 0,
            "vasc": 0
        }
        
        # Pre-calculate labels for weighted loss calculation later if needed
        self.labels = [self.label_map[dx] for dx in self.data['dx']]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = row['image_id'] + ".jpg"
        
        # Search in all directories
        img_path = None
        for d in self.img_dirs:
            potential_path = os.path.join(d, img_name)
            if os.path.exists(potential_path):
                img_path = potential_path
                break
        
        if img_path is None:
            raise FileNotFoundError(f"Image {img_name} not found in any of {self.img_dirs}")

        image = Image.open(img_path).convert("RGB")
        label = self.label_map[row['dx']]

        if self.transform:
            image = self.transform(image)

        return image, label

dataset = SkinCancerDataset(
    csv_file="../data/HAM10000_metadata.csv",
    img_dirs=["../data/HAM10000_images_part_1/", "../data/HAM10000_images_part_2/"],
    transform=None
)
BATCH_SIZE = 32


class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# ========================
# Train-Test Split
# ========================
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

# Split dataset into 80% train and 20% test, STRATIFIED
indices = list(range(len(dataset)))
train_indices, test_indices = train_test_split(indices, test_size=0.2, stratify=dataset.labels, random_state=42)

train_subset = Subset(dataset, train_indices)
test_subset = Subset(dataset, test_indices)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5), # Removed hue/saturation to preserve skin tones
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = TransformedSubset(train_subset, transform=train_transform)
test_dataset = TransformedSubset(test_subset, transform=test_transform)

# ========================
# Model Configuration
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2  # Binary classification: cancer (1) vs no cancer (0)
epochs = 80      

model = timm.create_model(
    "efficientformer_l1",
    pretrained=True,
    num_classes=num_classes,
    drop_rate=0.4,       # Dropout
    drop_path_rate=0.2    # Stochastic Depth
)

model.to(device)

# ========================
# Training Setup
# ========================
import numpy as np
from torch.utils.data import WeightedRandomSampler

# Calculate class weights for WeightedRandomSampler
train_labels = [dataset.labels[i] for i in train_dataset.subset.indices]
train_labels_arr = np.array(train_labels)
class_sample_count = np.array([len(np.where(train_labels_arr == t)[0]) for t in np.unique(train_labels_arr)])
weight = 1. / class_sample_count
samples_weight = np.array([weight[t] for t in train_labels])
samples_weight = torch.from_numpy(samples_weight).double()

sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=4 if os.name != 'nt' else 0,
    pin_memory=True
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4 if os.name != 'nt' else 0,
    pin_memory=True
)

# Label smoothing only. No class weights here because the sampler already balances the batches
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# OneCycleLR requires max_lr
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-4,
    epochs=epochs,
    steps_per_epoch=len(train_dataloader),
    pct_start=0.1 # 10% of training spent warming up
)

# ========================
# Training Loop
# ========================
best_test_acc = 0.0

for epoch in range(epochs):
    # Training phase
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step() # Step per batch for OneCycleLR

        train_loss += loss.item()
        
        # Calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_accuracy = 100 * train_correct / train_total
    avg_train_loss = train_loss / len(train_dataloader)

    # Testing/Evaluation phase
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Calculate test accuracy
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_accuracy = 100 * test_correct / test_total
    avg_test_loss = test_loss / len(test_dataloader)

    # Save the best model
    if test_accuracy > best_test_acc:
        best_test_acc = test_accuracy
        torch.save(model.state_dict(), "best_efficientformer_model.pth")
        save_msg = " (Best Model Saved!)"
    else:
        save_msg = ""

    print(f"Epoch {epoch+1}/{epochs} | "
          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
          f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%{save_msg}")

print(f"\nTraining completed! Best Test Accuracy: {best_test_acc:.2f}%")
torch.save(model.state_dict(), "last_efficientformer_model.pth")