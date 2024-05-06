# Import dependencies
import torch 
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


device = "cpu"
torch.set_default_device(device)
    
transform = transforms.Compose([transforms.Resize((224, 224)),  # Resize images to the same size
                                transforms.ToTensor()])

batch_size = 32
    
train = datasets.ImageFolder(root='data/train', transform=transform)
validation = datasets.ImageFolder(root='data/validation', transform=transform)

X = DataLoader(train, batch_size=batch_size, shuffle=True)
y = DataLoader(validation, batch_size=batch_size, shuffle=True)

class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32*112*112, 2)
        )

    def forward(self, x):
        return self.model(x)
    
clf = ImageClassifier().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(clf.parameters(), lr=1e-4)

if __name__ == "__main__":
    for epoch in range(3):
        for i, (images, labels) in enumerate(X):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = clf(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Batch {i}, Loss {loss.item()}")
            
    save(clf.state_dict(), "model.pth")
    clf.load_state_dict(load("model.pth"))
    clf.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in y:
            images, labels = images.to(device), labels.to(device)
            outputs = clf(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"Accuracy: {100 * correct / total}")
