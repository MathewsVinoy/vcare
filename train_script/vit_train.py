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

        # Map the 7 labels to 7 individual classes (0 to 6)
        self.label_map = {
            "mel": 0,   # Melanoma
            "bcc": 1,   # Basal Cell Carcinoma
            "akiec": 2, # Bowen’s / SCC in situ
            "nv": 3,
            "bkl": 4,
            "df": 5,
            "vasc": 6
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
num_classes = 7  # 7-class classification
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
# Overfitting Detector Config
# ========================
OVERFIT_PATIENCE      = 5    # consecutive epochs the signal must hold before we exit
OVERFIT_GAP_THRESHOLD = 15.0 # (%) train_acc - test_acc gap that flags overfitting
OVERFIT_LOSS_MIN_DELTA = 0.01 # min delta for test loss to be considered "rising"

overfit_streak = 0           # how many consecutive epochs overfitting signal fired

# ========================
# Training Loop
# ========================
best_test_acc  = 0.0
prev_test_loss = float("inf")

for epoch in range(epochs):
    # ── Training phase ─────────────────────────────────────────
    model.train()
    train_loss    = 0
    train_correct = 0
    train_total   = 0

    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()   # Step per batch for OneCycleLR

        train_loss    += loss.item()
        _, predicted   = torch.max(outputs.data, 1)
        train_total   += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_accuracy = 100 * train_correct / train_total
    avg_train_loss = train_loss / len(train_dataloader)

    # ── Evaluation phase ────────────────────────────────────────
    model.eval()
    test_loss    = 0
    test_correct = 0
    test_total   = 0

    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs  = model(images)
            loss     = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total   += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_accuracy  = 100 * test_correct / test_total
    avg_test_loss  = test_loss / len(test_dataloader)

    # ── Save best model ─────────────────────────────────────────
    if test_accuracy > best_test_acc:
        best_test_acc = test_accuracy
        torch.save(model.state_dict(), "best_efficientformer_model.pth")
        save_msg = " ✅ (Best Model Saved!)"
    else:
        save_msg = ""

    # ── Overfitting detection ────────────────────────────────────
    acc_gap        = train_accuracy - test_accuracy          # large gap = overfit
    loss_diverging = avg_test_loss > prev_test_loss + OVERFIT_LOSS_MIN_DELTA  # test loss rising

    overfit_signal = (acc_gap > OVERFIT_GAP_THRESHOLD) and loss_diverging

    if overfit_signal:
        overfit_streak += 1
    else:
        overfit_streak = 0   # reset if a single epoch looks healthy

    prev_test_loss = avg_test_loss

    print(f"Epoch {epoch+1}/{epochs} | "
          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
          f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}% | "
          f"Gap: {acc_gap:.2f}%  Overfit streak: {overfit_streak}/{OVERFIT_PATIENCE}"
          f"{save_msg}")

    # ── Exit on confirmed overfitting ────────────────────────────
    if overfit_streak >= OVERFIT_PATIENCE:
        print("\n" + "=" * 60)
        print("🚨 OVERFITTING DETECTED — stopping training early!")
        print(f"   Train Acc  : {train_accuracy:.2f}%")
        print(f"   Test  Acc  : {test_accuracy:.2f}%")
        print(f"   Gap        : {acc_gap:.2f}%  (threshold: {OVERFIT_GAP_THRESHOLD}%)")
        print(f"   Streak     : {overfit_streak} consecutive epochs")
        print(f"   Best model saved at epoch where test acc = {best_test_acc:.2f}%")
        print("=" * 60)
        torch.save(model.state_dict(), "last_efficientformer_model.pth")
        raise SystemExit(1)   # exit with error code so a calling script can detect it

print(f"\nTraining completed! Best Test Accuracy: {best_test_acc:.2f}%")
torch.save(model.state_dict(), "last_efficientformer_model.pth")