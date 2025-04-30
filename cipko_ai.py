import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageOps, ImageFilter
import matplotlib.pyplot as plt
import numpy as np

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load training config from JSON
with open("config.json", "r") as f:
    config = json.load(f)

# Config variables with sensible defaults
EPOCHS = config.get("epochs", 2)
LEARNING_RATE = config.get("learning_rate", 0.001)
HIDDEN_UNITS = config.get("hidden_units", 128)
CONV_CHANNELS = config.get("conv_channels", 16)
BATCH_SIZE = config.get("batch_size", 64)

# Define image transformation: Tensor + Normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset (digits 0–9)
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Neural Network model
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, CONV_CHANNELS, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(CONV_CHANNELS * 14 * 14, HIDDEN_UNITS)
        self.fc2 = nn.Linear(HIDDEN_UNITS, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, CONV_CHANNELS * 14 * 14)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Instantiate model, loss, and optimizer
model = DigitCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
print("🧠 Training the model...")
for epoch in range(EPOCHS):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

    print(f"✅ Epoch {epoch + 1}/{EPOCHS} - Loss: {loss.item():.4f}")

# Evaluate model on test set
model.eval()
correct = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        _, predicted = torch.max(output, 1)
        correct += (predicted == labels).sum().item()

print(f"📊 Test Accuracy: {correct / len(test_loader):.2%}")

# Predict digit from custom image
def predict_custom_image(path):
    if not os.path.exists(path):
        print("❌ File not found.")
        return

    # Preprocess image
    image = Image.open(path).convert("L").filter(ImageFilter.SHARPEN)
    image.thumbnail((20, 20), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), 255)
    canvas.paste(image, ((28 - image.width) // 2, (28 - image.height) // 2))

    # Invert background if needed
    if np.array(canvas).mean() > 127:
        canvas = ImageOps.invert(canvas)

    # Convert to tensor and run prediction
    tensor = transform(canvas).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        probs = F.softmax(output, dim=1)
        pred = torch.argmax(probs)
        confidence = probs[0][pred].item()

    # Show result
    plt.imshow(canvas, cmap="gray")
    plt.title(f"Predicted: {pred.item()} | Confidence: {confidence * 100:.2f}%")
    plt.axis("off")
    plt.show()

# CLI entry
if __name__ == "__main__":
    img_path = input("📸 Enter path to digit image (JPG/PNG/etc): ")
    predict_custom_image(img_path)
