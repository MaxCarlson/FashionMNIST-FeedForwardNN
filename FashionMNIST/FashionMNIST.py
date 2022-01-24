import torch
import copy
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
#device = torch.device('cpu')

# constant for classes
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

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

#x = trainset.train_data.numpy()
#y = trainset.train_labels.numpy()
#xp = copy.deepcopy(x)
#yp = copy.deepcopy(y)

# Pollute data
#for i, c in enumerate(classes):
#    idx = np.where(y == i)
#    vals = x[idx]
#    vals = vals[0:int(len(vals)/100)]
#    for j in range(len(classes)):
#        if j == i:
#            continue
#        xp = np.append(xp, vals, axis=0)
#        yp = np.append(yp, [j for _ in range(len(vals))])

class PollutedDataset(torch.utils.data.Dataset):
    def __init__(self, features, targets):
        self.features = torch.from_numpy(features).float()
        self.targets = torch.from_numpy(targets).long()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]

#trainset = PollutedDataset(xp, yp)

x = testset.test_data.numpy()
y = testset.test_labels.numpy()
xp = copy.deepcopy(x)

# Circular Shift right-shift test images by 2
#xp = np.roll(xp, 2, axis=2)
# Vertical Shift pixels of test images up by 2
xp = np.roll(xp, -2, axis=1)

testset = PollutedDataset(xp, y)



epochs      = 20
InputSize   = 784
Neurons     = 1024
batchSize   = 10
lr          = 0.001

# dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize,
 shuffle=True, num_workers=3)

testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize,
 shuffle=False, num_workers=3)




model = torch.nn.Sequential(
    torch.nn.Linear(InputSize, Neurons),
    torch.nn.ReLU(),
    torch.nn.Linear(Neurons, Neurons),
    torch.nn.ReLU(),
    torch.nn.Linear(Neurons, len(classes))
    )
#model = torch.nn.Sequential(
#    torch.nn.Linear(InputSize, Neurons),
#    torch.nn.Sigmoid(),
#    torch.nn.Linear(Neurons, Neurons),
#    torch.nn.Sigmoid(),
#    torch.nn.Linear(Neurons, len(classes))
#    )

model = model.to(device)
#summary(model, (batchSize, InputSize, Neurons))

loss_fn = torch.nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0)

if __name__ == '__main__':

    accPerEpoch = []
    for e in range(epochs):
        def run(loader, train):
            numCorrect = 0
            for i, (features, targets) in enumerate(loader):
    
                features = features.to(device)
                targets = targets.to(device)
                x = torch.reshape(features, (len(targets), 784))
                pred = model(x)

                # Calculate number of correct values
                #maxVals, maxIdx = torch.max(pred, 1)
                #numCorrect += (maxIdx == targets).sum().item()

                if train:
                    optim.zero_grad()
                    loss = loss_fn(pred, targets)
                    if i % 100 == 0:
                        print(f'Epoch {e+1}, loss: {loss.item()}')
                    loss.backward()
                    optim.step()
                else:
                    # Calculate number of correct values
                    maxVals, maxIdx = torch.max(pred, 1)
                    numCorrect += (maxIdx == targets).sum().item()
            return numCorrect
            
        run(trainloader, train=True)
        numCorrect = run(testloader, train=False)

        acc = numCorrect / len(testset)
        print(f'Accuracy for Epoch {e+1}: {acc}')
        accPerEpoch.append(acc)

    plt.plot(range(len(accPerEpoch)), accPerEpoch)
    plt.title(f'Accuracy vs. Epoch, 2 x 1024 FC layers, ReLU, lr={lr}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

            