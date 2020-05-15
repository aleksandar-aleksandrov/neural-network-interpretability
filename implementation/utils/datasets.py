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


def get_images_for_class(dataset, num_class):
    return dataset.data[dataset.targets == num_class]


def generate_mean_image(images):
    image_sum = torch.zeros(images[0].size())

    for image in images:
        image_sum += image
    return image_sum / len(images)


def generate_mean_image_for_class(dataset, num_class):
    images = get_images_for_class(dataset, num_class)

    return generate_mean_image(images)


def get_random_image(dataset_loader):
    dataiter = iter(dataset_loader)
    images, labels = dataiter.next()

    image = images[0].expand(1, 1, 28, 28)
    label = labels[0]

    return image, label