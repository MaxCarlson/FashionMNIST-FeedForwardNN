import torch
import torchvision

# datasets
trainset = torchvision.datasets.FashionMNIST('./data',
 download=True,
 train=True)
testset = torchvision.datasets.FashionMNIST('./data',
 download=True,
 train=False)
# dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
 shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
 shuffle=False, num_workers=2)
# constant for classes
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')
