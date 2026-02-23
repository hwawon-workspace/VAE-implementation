import torch
from torchvision import datasets, transforms
import gdown


class DatasetLoader:
    def __init__(self, dataset_name, transform=None, augment_transform=None):
        self.dataset_name = dataset_name
        self.transform = transform
        self.augment_transform = augment_transform

    def get_dataset(self):
        if self.dataset_name == 'MNIST':
            return self._load_mnist()
        elif self.dataset_name == 'cifar10':
            return self._load_cifar10()
        elif self.dataset_name == 'celeba':
            return self._load_celeba()
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")

    def _load_mnist(self):
        transform = self.transform if self.transform else self.get_default_transforms()
        train_dataset = datasets.MNIST(root='data/', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root='data/', train=False, transform=transform, download=True)
        return train_dataset, test_dataset

    def _load_cifar10(self):
        transform = self.transform if self.transform else self.get_default_transforms()
        augment_transform = self.augment_transform if self.augment_transform else self.get_augmentation_transforms()
        train_dataset = datasets.CIFAR10(root='data/', train=True, transform=augment_transform, download=True)
        test_dataset = datasets.CIFAR10(root='data/', train=False, transform=transform, download=True)
        return train_dataset, test_dataset

    def _load_celeba(self):
        transform = self.transform if self.transform else self.get_default_transforms()
        train_dataset = datasets.CelebA(root='data/', split='train', transform=transform, download=True)
        test_dataset = datasets.CelebA(root='data/', split='test', transform=transform, download=True)
        return train_dataset, test_dataset

    def get_default_transforms(self):
        if self.dataset_name == 'MNIST':
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        elif self.dataset_name == 'cifar10':
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        elif self.dataset_name == 'celeba':
            return transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def get_augmentation_transforms(self):
        if self.dataset_name == 'MNIST':
            return None
        elif self.dataset_name == 'cifar10':
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        elif self.dataset_name == 'celeba':
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])




if __name__ == "__main__":
    # Test MNIST loading
    mnist_loader = DatasetLoader(dataset_name='MNIST')
    train_dataset, test_dataset = mnist_loader.get_dataset()
    print("MNIST dataset loaded. Train size:", len(train_dataset), "Test size:", len(test_dataset))

    # Test CIFAR-10 loading
    cifar10_loader = DatasetLoader(dataset_name='cifar10')
    train_dataset, test_dataset = cifar10_loader.get_dataset()
    print("CIFAR-10 dataset loaded. Train size:", len(train_dataset), "Test size:", len(test_dataset))

    # Test CelebA loading
    celeba_loader = DatasetLoader(dataset_name='celeba')
    train_dataset, test_dataset = celeba_loader.get_dataset()
    print("CelebA dataset loaded. Train size:", len(train_dataset), "Test size:", len(test_dataset))