import torch
import torchvision
import torchvision.transforms as transforms


def get_mnist(path: str = './data', batch_size: int = 64):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    trainset = torchvision.datasets.MNIST(root=path, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    testset = torchvision.datasets.MNIST(root=path, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    return trainset, trainloader, testset, testloader
