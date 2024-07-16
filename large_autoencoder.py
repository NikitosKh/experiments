import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt

class LargeAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        # here we use Resnet as a starter
        resnet = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(2048, latent_dim)
        
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 256 * 14 * 14),
            nn.ReLU(),
            nn.Unflatten(1, (256, 14, 14)),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        encoded = encoded.view(encoded.size(0), -1)
        latent = self.fc(encoded)
        decoded = self.decoder(latent)
        return decoded

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, args):
    setup(rank, world_size)
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
    
    model = LargeAutoencoder(args.latent_dim).to(rank)
    model = DDP(model, device_ids=[rank])
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            imgs, _ = batch
            imgs = imgs.to(rank)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, imgs)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            if rank == 0:
                torch.save(model.state_dict(), f"autoencoder_dim{args.latent_dim}.pth")
    
    cleanup()
    return best_loss

def run_experiment(world_size, args):
    latent_dims = [32, 64, 128, 256, 512, 1024]
    losses = []
    
    for dim in latent_dims:
        args.latent_dim = dim
        print(f"Training autoencoder with latent dimension {dim}")
        best_loss = mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)[0]
        losses.append(best_loss)
    
    plt.figure(figsize=(10, 6))
    plt.plot(np.log(latent_dims), np.log(losses), marker='o')
    plt.xlabel('Log of Latent Dimension')
    plt.ylabel('Log of Best Loss')
    plt.title('Log of Best Loss vs Log of Latent Dimension')
    plt.grid(True)
    plt.savefig('loss_vs_dim.png')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--gpus', type=int, default=4)
    args = parser.parse_args()
    
    run_experiment(args.gpus, args)
