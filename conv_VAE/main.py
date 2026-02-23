import torch
from torch.optim import Adam
import models.vae_models as vae_models
from data.datasets import get_data_loaders
from utils.experiment import VAExperiment
from utils.test import VAEtest
from utils.filename import generate_filename
import yaml
import argparse

def main(config):
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 데이터 로드
    dataset_loader = get_data_loaders(
        dataset_name = config['data']['dataset'],
        batch_size = config['data']['batch_size'],
        val_split = config['data']['val_split']
    )
    
    train_loader, valid_loader, test_loader, image_shape = dataset_loader
    channels = image_shape[0]
    
    # 모델, 옵티마이저 설정
    model = getattr(vae_models, config['model']['model'])
    model = model(
        input_shape = image_shape, 
        latent_dim = config['model']['latent_dim'], 
        hidden_dims = config['model']['hidden_dims']
    ).to(device)
    optimizer = Adam(model.parameters(), lr = config['model']['learning_rate'])
    
    # 학습, 평가
    experiment = VAExperiment(model, device, channels, config['log']['model_save_dir'])
    for epoch in range(1, config['model']['epochs']+1):
        experiment.train(train_loader, optimizer, epoch)
        experiment.validate(valid_loader, epoch) 
    
    # 테스트
    filename = generate_filename(config)
    test = VAEtest(model, config['log']['model_save_dir'], device, channels, config['log']['image_save_dir'],filename)
    test.test(test_loader, save_image = True)

# def parse_args(config = None):
#     parser = argparse.ArgumentParser(description = 'VAE Training Script')
    
#     # 기본 설정
#     parser.add_argument('--config', type=str, default='configs/mnist_vae.yaml', help='Path to the config file')
#     parser.add_argument('--batch_size', type=int, help='Batch size for training')
#     parser.add_argument('--epochs', type=int, help='Number of epochs to train')
#     parser.add_argument('--lr', type=float, help='Learning rate')
#     parser.add_argument('--latent_dim', type=int, help='Dimensionality of the latent space')
#     parser.add_argument('--hidden_dims', type=list, help='Hidden layer dimensions')

#     args = parser.parse_args()

#     # config 파일 로드
#     with open(args.config, 'r') as file:
#         config = yaml.safe_load(file)

#     # argparse 인자를 config에 반영
#     if args.batch_size is not None:
#         config['data']['batch_size'] = args.batch_size
#     if args.epochs is not None:
#         config['model']['epochs'] = args.epochs
#     if args.lr is not None:
#         config['model']['learning_rate'] = args.lr
#     if args.latent_dim is not None:
#         config['model']['latent_dim'] = args.latent_dim
#     if args.hidden_dims is not None:
#         config['model']['hidden_dims'] = args.hidden_dims

#     return config

    
if __name__ == "__main__":
    try:
        # Jupyter Notebook 환경에서는 argparse를 사용하지 않음
        import sys
        if "ipykernel_launcher" in sys.argv[0]:
            # config_path = 'configs/mnist_vae.yaml'
            # config_path = 'configs/fmnist_vae.yaml'
            config_path = 'configs/mnist_convvae.yaml'
        else:
            args = parse_args()
            config_path = args.config
        
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        main(config)
    
    except Exception as e:
        print(f"Error: {e}")

