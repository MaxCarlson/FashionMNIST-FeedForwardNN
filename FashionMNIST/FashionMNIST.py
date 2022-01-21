import numpy
import torch
import torchvision
from torchvision.transforms import ToTensor

# datasets
trainset = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=True,
    transform=ToTensor()
    )
testset = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=False,
    transform=ToTensor()
    )

InputSize = 784
Neurons = 1024
batchSize = 30
epochs = 2
lr = 0.001

# dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize,
 shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize,
 shuffle=False, num_workers=2)

# constant for classes
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')



model = torch.nn.Sequential(
    torch.nn.Linear(InputSize, Neurons),
    torch.nn.ReLU(),
    torch.nn.Linear(Neurons, len(classes))
    )

loss_fn = torch.nn.CrossEntropyLoss()

if __name__ == '__main__':

    for e in range(epochs):
        for train_features, train_labels in trainloader:
    
            a = train_features.numpy()
            x = torch.reshape(train_features, (30, 784))
            a = x.numpy()

            pred = model(x)
    
            loss = loss_fn(pred, train_labels)
            model.zero_grad()
            loss.backward()

            with torch.no_grad():
                for param in model.parameters():
                    param -= lr * param.grad