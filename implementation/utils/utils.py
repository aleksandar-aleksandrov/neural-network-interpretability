import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_examples(images):
    imshow(torchvision.utils.make_grid(images))


def show_dataset_examples(trainloader):
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    images = images[0:4]
    labels = labels[0:4]

    print(' '.join('%5s' % labels[j].item() for j in range(4)))

    show_examples(images)


def show_neighbors(left_image, right_image):
    show = torch.zeros((2, 1, 28, 28))
    show[0] = left_image
    show[1] = right_image

    imshow(torchvision.utils.make_grid(show))
