import torch
import matplotlib.pyplot as plt

from torchvision import datasets, transforms


def read_train(kwargs):

    # Train data transformations
    train_transforms = transforms.Compose([
        transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
        transforms.Resize((28, 28)),
        transforms.RandomRotation((-15., 15.), fill=0),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        ])
    
    train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_data, **kwargs)

    return train_loader

def read_test(kwargs):
    # Test data transformations
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_data, **kwargs)

    return test_loader


def show_images(train_loader):
    fig = plt.figure()
    batch_data, batch_label = next(iter(train_loader)) 

    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0), cmap='gray')
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])


def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()