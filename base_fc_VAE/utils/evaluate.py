import torch
import torch.nn.functional as F

def evaluate(model, dataloader, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)            
            # Forward pass
            recon_batch, mu, logvar = model(data)            
            # Compute loss
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
    avg_loss = test_loss / len(dataloader.dataset)
    print(f"====> Test set loss: {avg_loss:.4f}")
    return avg_loss

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
