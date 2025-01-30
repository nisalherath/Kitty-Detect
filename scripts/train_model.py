import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import ResNet50_Weights
from torch.amp import GradScaler, autocast
from tqdm import tqdm

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
def train_model(data_dir, model_save_path):
    print("Training model...")

    # Define transformations with enhanced data augmentation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),  # Slight translation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),  # Slight blur
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard ImageNet normalization
    ])

    # Load dataset and split into train/validation (80%-20%)
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Define the model (using a pre-trained ResNet50)
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)  # Assuming 2 classes (my_cat and other_cats)

    # Move model to device (GPU if available)
    model.to(device)

    # Freeze early layers and fine-tune deeper layers
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():  # Unfreeze the last few layers
        param.requires_grad = True

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Use cosine annealing scheduler for smooth learning rate adjustments
    scheduler = CosineAnnealingLR(optimizer, T_max=10)

    # Mixed precision training
    scaler = GradScaler(enabled=torch.cuda.is_available())  # Enable only if CUDA is available

    # Early stopping parameters
    best_loss = float('inf')
    patience = 3
    no_improvement = 0

    # Train the model
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Progress bar for training loop
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Training", ncols=100)

        for images, labels in train_progress:
            images, labels = images.to(device), labels.to(device)  # Move data to GPU if available
            optimizer.zero_grad()

            with autocast(device_type=device.type):  # Corrected autocast usage
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            # Update progress bar description with loss
            train_progress.set_postfix(loss=running_loss / (train_progress.n + 1))

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # Learning rate scheduling
        scheduler.step()

        # Validation accuracy tracking
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0

        # Progress bar for validation loop
        val_progress = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Validation", ncols=100)

        with torch.no_grad():
            for images, labels in val_progress:
                images, labels = images.to(device), labels.to(device)  # Move to GPU if available
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                # Update progress bar description with loss
                val_progress.set_postfix(loss=val_loss / (val_progress.n + 1))

        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total

        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            no_improvement = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"✅ Model saved to {model_save_path}")
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"⏹️ No improvement for {patience} epochs. Stopping training.")
                break

    print("🎉 Training complete.")
