import torch
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms

# 데이터셋 모듈화
DATASETS_INFO = {
    # MNIST
    "MNIST" : {
        "image_shape": (1, 28, 28), # (C, H, W)
        "num_classes": 10,
        "data_path": "./data/mnist",
        "transform": transforms.Compose([
            transforms.ToTensor(),
            # mean = 0.1307, std = 0.3081
        ])
    },
    
    # CIFAR10
    "CIFAR10" : {
        "image_shape": (3, 32, 32), # (C, H, W)
        "num_classes": 10,
        "data_path": "./data/cifar10",
        "transform": transforms.Compose([
            transforms.ToTensor(),
            # mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  
        ])
    },
    
    # FASHION_MNIST
    "FASHION_MNIST" : {
        "image_shape": (1, 28, 28), # (C, H, W)
        "num_classes": 10,
        "data_path": "./data/fashion_mnist",
        "transform": transforms.Compose([
            transforms.ToTensor(),
        ])
    },
    
    # SVHN
    "SVHN" : {
        "image_shape": (3, 32, 32), # (C, H, W)
        "num_classes": 10,
        "data_path": "./data/svhn",
        "transform": transforms.Compose([
            transforms.ToTensor(),
            # mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970]
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))  
        ])
    },
    
    # CelebA
    "CelebA" : {
        "image_shape": (3, 64, 64), # (C, H, W)
        "num_classes": None,
        "data_path": "./data/celeba",
        "transform": transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            # mean=[0.5063, 0.4258, 0.3832], std=[0.2657, 0.2457, 0.2412]
            transforms.Normalize((0.5063, 0.4258, 0.3832), (0.2657, 0.2457, 0.2412))
        ])
    },
}

# 데이터셋 모듈 불러오기
def get_dataset_info(dataset_name):
    return DATASETS_INFO.get(dataset_name)

# 데이터셋 로드
def get_data_loaders(dataset_name, batch_size, val_split=0.1):
    # dataset_info 불러오기
    dataset_info = get_dataset_info(dataset_name)
    
    # 정의된 데이터셋 모듈 내에 없을 경우
    if dataset_info is None:
        raise ValueError(f"Dataset {dataset_name} is not defined in DATASETS_INFO.")
    
    # 데이터셋 맞춤 transform, data path
    transform = dataset_info['transform']
    data_path = dataset_info['data_path']
    
    # 데이터셋 클래스를 매핑해 데이터셋 로드
    dataset_class = {
        "MNIST": datasets.MNIST,
        "CIFAR10": datasets.CIFAR10,
        "FASHION_MNIST": datasets.FashionMNIST,
        "SVHN": datasets.SVHN,
        "CelebA": datasets.CelebA
    }.get(dataset_name)
    
    if dataset_class is None:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    # Train/Valid/Test 데이터셋
    if dataset_name in ['SVHN', 'CelebA']:
        train_dataset = dataset_class(root = data_path, split = 'train', transform = transform, download = True)
        test_dataset = dataset_class(root = data_path, split = 'test', transform = transform, download = True)
    else:
        train_dataset = dataset_class(root = data_path, train = True, transform = transform, download = True)
        test_dataset = dataset_class(root = data_path, train = False, transform = transform, download = True)
    
    val_size = int(len(train_dataset) * val_split)
    train_size = int(len(train_dataset) * (1 - val_split))
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    return train_loader, val_loader, test_loader, dataset_info['image_shape']