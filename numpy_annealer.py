import torch.nn as nn
import torch.nn.functional as f
from torchinfo import summary
from torchvision import models, datasets
from torchvision import transforms
import torchvision
import torch
import torch.optim as optim
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import pyqubo

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

# normal
def train_loop(model, dataloader, optimizer, criterion, num_epochs, cutout=None):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            if cutout and i > cutout:
                break
            X, y = data
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000}')
                running_loss = 0.0
    print("Finished training")

def forward_no_fc(model, x):
    """
    Forward pass used to calculate QUBO matrix.
    """

    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    # x = model.fc(x)

    return x


def calculate_pyqubo(model, X, y):
    """
    Input: model and batch from dataloader.
    Add result from all images and call .compile().
    Make sure the last layer of the model is fully connected and named fc.
    """
    H = 0
    # Extract one example from batch.
    # for X, y in zip(images, labels):
    # Calculate result before fully connected layer.
    x = forward_no_fc(model, X)
    for input, label in zip(x, y):
        labels = np.zeros(10)
        labels[label] = 1
        # Assign a qubit to every of the layer.
        x = np.array([pyqubo.Binary(f"{i}") * input[i] for i in range(len(input))])
        # Calculate output in terms of QUBO.
        y_pred = model.fc.weight.detach().numpy() @ x
        # Add to hamiltonian after calculating squared error.
        H += ((labels-y_pred)**2).sum()
    return H


def anneal_loop(model, dataloader):
    """
    Loops over all images in dataloader adding to Hamiltonian.
    """
    outputs = []
    expecteds = []

    model.train()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            images, labels = data
            a = forward_no_fc(model, images)
            outputs.extend(a)
            expected = torch.eye(model.fc.out_features)[labels]
            expecteds.extend(expected)

            # H += calculate_pyqubo(model, images, labels)
            # if i%100 == 0:
            print(f"{i+1} / {len(dataloader)}")
        return torch.stack(outputs), torch.stack(expecteds)


def test(model, dataloader):
    model.eval()
    running_accuracy = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            X, y = data
            y_pred = model(X)
            _, predicted = torch.max(y_pred, 1)
            total += y.size(0)
            running_accuracy += (predicted == y).sum().item()
        print('Accuracy of the model based on the test set of','all','inputs is: %d %%' % (100 * running_accuracy / total))



if __name__ == '__main__':
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # model.forward()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)



    # feature extraction
    for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(512,10, bias=False) # reset last layer for fine-tuning

    # datasets
    datasets = {
        name: datasets.MNIST(
            root='.', download=True, train=(name=='train'), transform=transforms.Compose([
                transforms.ToTensor()
            ])
        ) for name in ['train', 'val']
    }

    # dataloaders
    class_names = datasets['train'].classes
    dataloaders = {
        name: torch.utils.data.DataLoader( #pyright: ignore
            datasets[name], batch_size=4, shuffle=True
        ) for name in ['train', 'val']
    }
    dataloaders['anneal'] = torch.utils.data.DataLoader(datasets['train'], batch_size=1000, shuffle=False) #pyright: ignore
    # training loop
    num_epochs = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.7)

    # turn off gradient for last layer
    for param in model.fc.parameters():
        param.requires_grad = False

    # train_loop(model, dataloaders['train'], optimizer, criterion, num_epochs, cutout=1500)
    outputs, expecteds = anneal_loop(model, dataloaders['anneal'])


    # inputs, classes = next(iter(dataloaders['train']))
    # summary(model)

    # Make a grid from batch
    # out = torchvision.utils.make_grid(inputs)

    # imshow(out, title=[class_names[x] for x in classes])
    # plt.show()

