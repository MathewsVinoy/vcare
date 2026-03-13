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
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = row['image_id'] + ".jpg"  # Change if png/jpeg differs
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        label = self.label_map[row['dx']]

        if self.transform:
            image = self.transform(image)

        return image, label

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = SkinCancerDataset(
    csv_file="../data/HAM10000_metadata.csv",
    img_dir="../data/HAM10000_images_part_1/",  # path to folder with .jpg images
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
from torch.utils.data import DataLoader, random_split

# Split dataset into 80% train and 20% test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataset = TransformedSubset(train_dataset, transform=train_transform)
test_dataset = TransformedSubset(test_dataset, transform=test_transform)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ========================
# Model Configuration
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2  # Binary classification: cancer (1) vs no cancer (0)
lr = 1e-4
epochs = 10

model = timm.create_model(
    "efficientformer_l1",
    pretrained=True,
    num_classes=num_classes
)

model.to(device)

# ========================
# Training Setup
# ========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

# ========================
# Training Loop
# ========================
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
        optimizer.step()

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

    scheduler.step()

    print(f"Epoch {epoch+1}/{epochs} | "
          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
          f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")

print("\nTraining completed!")
torch.save(model.state_dict(), "efficientformer_model.pth")
print("Model saved as 'efficientformer_model.pth'")