import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import numpy as np
import time

from logger import *

criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()

# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 3) # Cambio de 3 a 1 (rgb -> b/n)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 7, 3)
        self.fc1 = nn.Linear(7 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2) # Cambio de 10 a 2

    def forward(self, x):
        # print(self.conv1(x).shape)
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.shape)
        
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x)        # Para facilitar luego el entrenamiento
        return x

### Template

def train(x, y, net, epochs, lr, verbose=True, **kwargs):
    opt = torch.optim.Adam(net.parameters(), lr)
    # opt = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    loss = nn.BCELoss()

    for it in range(epochs):
        # for i in range(139):
        opt.zero_grad()

        y_hat = net(x)
        loss_eval = loss(y_hat[:, 1], y)    # y_hat[:, 1]
        loss_eval.backward()

        _, y_hat = torch.max(y_hat, 1)
        accuracy = np.mean(np.where(y_hat.detach().numpy() == y.detach().numpy(), 1, 0))

        if kwargs:
            y_hat_test = net(kwargs['x_test'])
            _, y_hat_test = torch.max(y_hat_test, 1)
            accuracy_test = np.mean(np.where(y_hat_test.detach().numpy() == kwargs['y_test'].detach().numpy(), 1, 0))

            wandb.log({'accuracy_test': accuracy_test})

        wandb.log({'loss': loss_eval, 'accuracy': accuracy})

        opt.step()

        if verbose:
            if it % 5 == 0:
                print(f"Loss at step {it}: {loss_eval}")

    return loss

def run_torch_cnn(dataset, epochs, lr, verbose=False):
    # build the circuit
    # circuit = get_circuit(qcnn, device="default.qubit")
    y_train_tensor = torch.tensor(dataset.y_train, dtype=torch.float)
    x_train_tensor = torch.tensor(np.reshape(dataset.x_train, (dataset.x_train.shape[0], 8, 32))[:, np.newaxis, ...], dtype=torch.float)

    y_test_tensor = torch.tensor(dataset.y_test, dtype=torch.float)
    x_test_tensor = torch.tensor(np.reshape(dataset.x_test, (dataset.x_test.shape[0], 8, 32))[:, np.newaxis, ...], dtype=torch.float)

    # print(x_train_tensor.shape)
    net = Net()
    wandb.run.config["parameters"] = sum(p.numel() for p in net.parameters())

    # train cnn
    t0 = time.time()
    loss = train(x_train_tensor, y_train_tensor, net, epochs, lr, verbose=verbose, x_test=x_test_tensor, y_test=y_test_tensor)
    trtime = time.time() - t0
    
    # get predictions
    y_hat = net(torch.tensor(np.reshape(dataset.x_test, (dataset.x_test.shape[0], 8, 32))[:, np.newaxis, ...], dtype=torch.float))

    # evaluate
    # y_hat = torch.argmax(y_hat, axis=1).detach().numpy()
    _, y_hat = torch.max(y_hat, 1)

    # y_hat = torch.round(y_hat).detach().numpy()

    accuracy = np.mean(np.where(y_hat.detach().numpy() == dataset.y_test, 1, 0))

    wandb.run.summary["test_accuracy"] = accuracy

    return accuracy, trtime