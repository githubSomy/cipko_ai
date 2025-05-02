import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np

# Load config
with open("config.json", "r") as f:
    config = json.load(f)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config vars
EPOCHS = config.get("epochs", 5)
LEARNING_RATE = config.get("learning_rate", 0.001)
HIDDEN_UNITS = config.get("hidden_units", 128)
CONV_CHANNELS = config.get("conv_channels", 16)
CONV_LAYERS = config.get("conv_layers", 2)
DROPOUT = config.get("dropout", 0.25)
BATCHNORM = config.get("batchnorm", True)
BATCH_SIZE = config.get("batch_size", 64)
SHARPEN = config.get("sharpen", True)
INVERT_THRESHOLD = config.get("invert_threshold", 127)

MODEL_PATH = "digit_cnn.pth"

# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Data
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# CNN model
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, CONV_CHANNELS, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(CONV_CHANNELS)
        self.conv2 = nn.Conv2d(CONV_CHANNELS, CONV_CHANNELS * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(CONV_CHANNELS * 2)
        self.pool = nn.MaxPool2d(2, 2)

        # 🔥 Use dummy input to calculate flattened feature size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 28, 28)
            x = self.pool(F.relu(self.bn1(self.conv1(dummy_input))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            self.flattened_size = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flattened_size, HIDDEN_UNITS)
        self.fc2 = nn.Linear(HIDDEN_UNITS, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Instantiate
model = DigitCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("📥 Loaded model from disk.")
else:
    print("🧠 Training...")
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
    torch.save(model.state_dict(), MODEL_PATH)
    print("💾 Saved model.")

# Evaluation
model.eval()
correct = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        _, predicted = torch.max(output, 1)
        correct += (predicted == labels).sum().item()
print(f"📊 Test Accuracy: {correct / len(test_loader):.2%}")

# Prediction on custom image
def predict_custom_image(path):
    if not os.path.exists(path):
        print("❌ File not found.")
        return

    # Load and enhance image
    image = Image.open(path).convert("L")
    if SHARPEN:
        image = image.filter(ImageFilter.SHARPEN)
    image = ImageEnhance.Contrast(image).enhance(2.0)
    image.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # Paste onto 28x28 canvas
    canvas = Image.new("L", (28, 28), 255)
    canvas.paste(image, ((28 - image.width) // 2, (28 - image.height) // 2))

    # Optional inversion
    if np.array(canvas).mean() > INVERT_THRESHOLD:
        canvas = ImageOps.invert(canvas)

    # To tensor
    tensor = transform(canvas).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        probs = F.softmax(output, dim=1)
        pred = torch.argmax(probs)
        confidence = probs[0][pred].item()

    # Show
    plt.imshow(canvas, cmap="gray")
    plt.title(f"Predicted: {pred.item()} | Confidence: {confidence * 100:.2f}%")
    plt.axis("off")
    plt.show()

# CLI
if __name__ == "__main__":
    img_path = input("📸 Enter path to digit image: ")
    predict_custom_image(img_path)
