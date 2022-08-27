import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch import nn
from tqdm import tqdm
from dimod import BinaryQuadraticModel, ExactSolver
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridBQMSampler
from neal import SimulatedAnnealingSampler
from copy import deepcopy
import dwave.inspector

last_layer_size = 100

class Network(nn.Module):
    def __init__(self) -> None:
        super(Network, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, last_layer_size, bias=True),
            nn.ReLU(),
            nn.Linear(last_layer_size, 2, bias=False),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

    def forward_no_fc(self, x):
        """
        Forward pass used to calculate QUBO matrix.
        """
        x = self.linear_relu_stack[0](x)
        x = self.linear_relu_stack[1](x)
        # x = self.linear_relu_stack[2](x)
        # x = self.linear_relu_stack[3](x)
        return x


class XORDataset(Dataset):
    def __init__(self) -> None:
        self.X = torch.tensor([(0.,0.), (0.,1.), (1.,0.), (1.,1.)], dtype=torch.float)
        self.y = torch.tensor([(1,0),(0,1),(0,1),(1,0)], dtype=torch.float)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


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
            a = model.forward_no_fc(images)
            # print(f"{a=}")
            outputs.extend(a)
            print(model.linear_relu_stack[-1].out_features)
            # print(f"{labels=}")
            # expected = torch.eye(model.linear_relu_stack[-1].out_features)[labels.to(torch.long)]
            expected = labels
            # print(f"{expected=}")
            expecteds.extend(expected)

            # H += calculate_pyqubo(model, images, labels)
            # if i%100 == 0:
            print(f"{i+1} / {len(dataloader)}")
        outputs, expecteds = torch.stack(outputs), torch.stack(expecteds)
        return calculate_qubo_matrix(model, outputs, expecteds)

def calculate_qubo_matrix(model, outputs, expecteds):
    """
    Input: model and batch from dataloader.
    Add result from all images and call .compile().
    Make sure the last layer of the model is fully connected and named fc.

    no bias for now

    outputs: A
    expecteds: Y'
    model.fc.weights: W
    """
    W = model.linear_relu_stack[-1].weight.detach().numpy()
    A = outputs.detach().numpy()
    # print(f"{W=}")
    # print(f"{A=}")
    Y = expecteds.detach().numpy()
    # Q = torch.zeros(model.fc.in_features, model.fc.in_features)
    print('zaczynam einsum')
    Q = np.einsum('di,ei,dj,ej->ij',W,A,W,A)
    # print(Q)
    np.fill_diagonal(Q,0)
    print('Calculating Q(i,i):')
    for i in tqdm(range(W.shape[1])):
        for e in range(A.shape[0]):
            for d in range(W.shape[0]):
                Q[i,i] += (W[d,i]*A[e,i])**2 - 2*W[d,i]*A[e,i]*Y[e,d]
    # print(Q)
    return BinaryQuadraticModel(Q, "BINARY")

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
            # print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000}')
    print("Finished training")

def test(model, dataloader):
    model.eval()
    running_accuracy = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            X, y = data
            predicted = model(X)
            print(X, y, predicted)
            total += y.size(0)
            predicted_onehot = torch.zeros_like(predicted)
            predicted_onehot[0][predicted.argmax()] = 1
            running_accuracy += (predicted_onehot == y).sum().item()
            print(running_accuracy/total)
        print('Accuracy of the model based on the test set of','all','inputs is: %d %%' % (100 * running_accuracy / total))

if __name__ == '__main__':
    model = Network()
    dataset = XORDataset()
    for X, y in dataset:
        print(X, y)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    # training loop
    num_epochs = 1000

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.7)
    test(model, dataloader)
    train_loop(model, dataloader, optimizer, criterion, num_epochs)
    test(model, dataloader)
    trained_model = deepcopy(model)

    # model.linear_relu_stack[-1] = nn.Linear(last_layer_size, 2, bias=False)
    # model.linear_relu_stack[-1].reset_parameters()
    torch.nn.init.kaiming_normal_(model.linear_relu_stack[0].weight)
    torch.nn.init.kaiming_normal_(model.linear_relu_stack[-1].weight)

    # turn off gradient for last layer
    for param in model.linear_relu_stack[0].parameters():
        param.requires_grad = False
    for param in model.linear_relu_stack[-1].parameters():
        param.requires_grad = False

    w = torch.tensor([[ 0.4155,  0.8025],
                [-0.2612, -0.8669],
                [-0.8477, -0.0145],
                [-1.2908, -1.2908]], requires_grad=False)
    b = torch.tensor([0.3247, 1.1282, 0.9496, 1.2908], requires_grad=False)
    w2 = torch.tensor([[ 0.6870, -0.5059, -0.6868,  1.5493],
                 [-0.0464,  1.0969,  0.8189, -1.5496]], requires_grad=False)

    # model.linear_relu_stack[0].weight[0:4,0:2] = w
    # model.linear_relu_stack[0].bias[0:4] = b
    # model.linear_relu_stack[2].weight[0:2,0:4] = w2


    test(model, dataloader)
    bqm = anneal_loop(model, dataloader)
    # print(bqm)
    sampleset = SimulatedAnnealingSampler().sample(bqm, num_reads=10000)
    # sampleset = EmbeddingComposite(DWaveSampler()).sample(bqm, num_reads=10000)
    # sampleset = LeapHybridBQMSampler().sample(bqm)
    # sampleset = ExactSolver().sample(bqm)
    print(sampleset.first.sample.values())
    print(sampleset.first.energy)
    dwave.inspector.show(sampleset)

    model.linear_relu_stack[-1].weight *= torch.tensor(list(sampleset.first.sample.values()))
    # setattr()

    test(model, dataloader)


# trained_model.linear_relu_stack[0].weight
# tensor([[ 0.4155,  0.8025],
#         [-0.2612, -0.8669],
#         [-0.8477, -0.0145],
#         [-1.2908, -1.2908]], requires_grad=True)
# trained_model.linear_relu_stack[0].bias
# tensor([0.3247, 1.1282, 0.9496, 1.2908], requires_grad=True)
#
#
# trained_model.linear_relu_stack[2].weight
# tensor([[ 0.6870, -0.5059, -0.6868,  1.5493],
#         [-0.0464,  1.0969,  0.8189, -1.5496]], requires_grad=True)
#
