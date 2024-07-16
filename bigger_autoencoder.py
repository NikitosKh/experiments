import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_autoencoder(model, train_loader, test_loader, optimizer, criterion, epochs):
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        for data in train_loader:
            img, _ = data
            img = img.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, img)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in test_loader:
                img, _ = data
                img = img.to(device)
                output = model(img)
                val_loss += criterion(output, img).item()
        val_loss /= len(test_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
        else:
            break

    return best_val_loss
latent_dims = [2, 4, 8, 16, 32, 64, 128]
val_losses = []
for dim in latent_dims:
    print(f"Training autoencoder with latent dimension {dim}")
    model = Autoencoder(dim).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    best_loss = train_autoencoder(model, train_loader, test_loader, optimizer, criterion, epochs=50)
    val_losses.append(best_loss)
plt.figure(figsize=(10, 6))
plt.plot(np.log(latent_dims), np.log(val_losses), marker='o')
plt.xlabel('Log of Latent Dimension')
plt.ylabel('Log of Validation Loss')
plt.title('Log of Validation Loss vs Log of Latent Dimension')
plt.grid(True)
plt.show()
def visualize_reconstruction(model, image, device):
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        reconstructed = model(image)

   
    image = image.cpu().squeeze().permute(1, 2, 0).numpy()
    reconstructed = reconstructed.cpu().squeeze().permute(1, 2, 0).numpy()

    
    image = (image * 0.5 + 0.5).clip(0, 1)
    reconstructed = (reconstructed * 0.5 + 0.5).clip(0, 1)

    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')

    ax2.imshow(reconstructed)
    ax2.set_title('Reconstructed Image')
    ax2.axis('off')

    plt.show()
random_index = torch.randint(0, 4, (1,)).item()
image, _ = train_dataset[4]


visualize_reconstruction(model, image, device)
