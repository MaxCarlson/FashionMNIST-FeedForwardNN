import torch
import torchvision
import numpy as np
from torchvision.transforms import ToTensor
from torchsummary import summary
import matplotlib.pyplot as plt

if torch.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu"  
device = torch.device(dev)  

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

epochs      = 30
InputSize   = 784
Neurons     = 1024
batchSize   = 30
lr          = 0.001

# dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize,
 shuffle=True, num_workers=4)

testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize,
 shuffle=False, num_workers=4)

# constant for classes
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')



model = torch.nn.Sequential(
    torch.nn.Linear(InputSize, Neurons),
    torch.nn.ReLU(),
    torch.nn.Linear(Neurons, Neurons),
    torch.nn.ReLU(),
    torch.nn.Linear(Neurons, len(classes))
    )

model = model.to(device)
#summary(model, (batchSize, InputSize, Neurons))

loss_fn = torch.nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0)

if __name__ == '__main__':

    accPerEpoch = []

    for e in range(epochs):
        def run(loader, train):
            acc = 0
            for i, (features, target) in enumerate(loader):
    
                features = features.to(device)
                target = target.to(device)
                x = torch.reshape(features, (len(target), 784))
                pred = model(x)

                # Calculate number of correct values
                #maxVals, maxIdx = torch.max(pred, 1)
                #acc += (maxIdx == target).sum().item()

                if train:
                    optim.zero_grad()
                    loss = loss_fn(pred, target)
                    if i % 100 == 0:
                        print(f'Epoch {e+1}, loss: {loss.item()}')
                    loss.backward()
                    optim.step()
                else:
                    # Calculate number of correct values
                    maxVals, maxIdx = torch.max(pred, 1)
                    acc += (maxIdx == target).sum().item()
            return acc
            
        run(trainloader, train=True)
        acc = run(testloader, train=False)

        acc /= len(testset)
        print(f'Accuracy for Epoch {e+1}: {acc}')
        accPerEpoch.append(acc)

    plt.plot(range(len(accPerEpoch)), accPerEpoch)
    plt.title('Accuracy vs. Epoch, 2 x 1024 FC layer')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

            