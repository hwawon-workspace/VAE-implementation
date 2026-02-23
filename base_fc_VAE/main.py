import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from models.VAE_MNIST import VAE_MNIST
from data.datasets import DatasetLoader
from utils.train import train
from utils.evaluate import evaluate
import yaml
import argparse

def main(config):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare dataset
    dataset_loader = DatasetLoader(
        dataset_name=config['data']['dataset'],
        transform=None,  # Use default transformation
        augment_transform=None  # Use default augmentation if applicable
    )    
    train_dataset, test_dataset = dataset_loader.get_dataset()
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    # Initialize model, optimizer
    model = VAE_MNIST(input_height=config['model']['input_height'], input_width=config['model']['input_width'], color_channels=config['model']['color_channels'], latent_dim=config['model']['latent_dim'], hidden_dims=config['model']['hidden_dims']).to(device)
    optimizer = Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=float(config['training']['weight_decay']))

    # Training and evaluation loop
    for epoch in range(1, config['training']['num_epochs'] + 1):
        train_loss = train(model, train_loader, optimizer, epoch, device, config['logging']['log_interval'])
        if epoch % config['evaluation']['evaluate_interval'] == 0:
            test_loss = evaluate(model, test_loader, device)

        # Save model
        if config['logging']['save_model'] and epoch == config['training']['num_epochs']:
            torch.save(model.state_dict(), config['logging']['model_save_path'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    
    parser.add_argument('--config', type=str, default='configs/VAE_MNIST.yaml', help='Path to the config file')
    parser.add_argument('--batch_size', type=int, help='Input batch size for training')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--latent_dim', type=int, help='Dimension of latent space')
    parser.add_argument('--log_interval', type=int, help='How many batches to wait before logging training status')
    
    args = parser.parse_args()
    
    # Load configuration from YAML file
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Override config parameters with command line arguments if provided
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.latent_dim:
        config['model']['latent_dim'] = args.latent_dim
    if args.log_interval:
        config['logging']['log_interval'] = args.log_interval

    main(config)
