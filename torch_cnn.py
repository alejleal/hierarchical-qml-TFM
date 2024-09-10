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
    def __init__(self, size, filters=(3,7), linear_sizes=(120, 84)):
        super().__init__()
        k1, k2 = 3, 3       # kernel sizes
        c1, c2 = filters       # convolution filters
        red_dim = lambda x: (x - k1 + 1)//2 - k2 + 1
        in_linear = red_dim(size[0])*red_dim(size[1])

        l1, l2 = linear_sizes
        # print(in_linear, red_dim(size[0]), red_dim(size[1]))

        self.conv1 = nn.Conv2d(1, c1, k1) # Cambio de 3 a 1 (rgb -> b/n)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(c1, c2, k2)
        self.fc1 = nn.Linear(c2 * in_linear, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 2) # Cambio de 10 a 2

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

            wandb.log({'accuracy_test': accuracy_test, 'loss': loss_eval, 'accuracy': accuracy})
        else:
            wandb.log({'loss': loss_eval, 'accuracy': accuracy})

        opt.step()

        if verbose:
            if it % 5 == 0:
                print(f"Loss at step {it}: {loss_eval}")

    return loss

def run_torch_cnn(dataset, epochs, lr, filters, linear_sizes, verbose=False):
    # build the circuit
    # circuit = get_circuit(qcnn, device="default.qubit")
    h, w = 8, 32

    y_train_tensor = torch.tensor(dataset.y_train, dtype=torch.float)
    x_train_tensor = torch.tensor(np.reshape(dataset.x_train, (dataset.x_train.shape[0], h, w))[:, np.newaxis, ...], dtype=torch.float)

    y_test_tensor = torch.tensor(dataset.y_test, dtype=torch.float)
    x_test_tensor = torch.tensor(np.reshape(dataset.x_test, (dataset.x_test.shape[0], h, w))[:, np.newaxis, ...], dtype=torch.float)

    net = Net((h, w), filters, linear_sizes)
    num_params = sum(p.numel() for p in net.parameters())
    wandb.run.config["parameters"] = num_params
    print(f"###\n Num. params: {num_params}\n###")

    # train cnn
    t0 = time.time()
    loss = train(x_train_tensor, y_train_tensor, net, epochs, lr, verbose=verbose, x_test=x_test_tensor, y_test=y_test_tensor)
    trtime = time.time() - t0
    
    # get predictions
    y_hat = net(torch.tensor(np.reshape(dataset.x_test, (dataset.x_test.shape[0], h, w))[:, np.newaxis, ...], dtype=torch.float))

    # evaluate
    # y_hat = torch.argmax(y_hat, axis=1).detach().numpy()
    _, y_hat = torch.max(y_hat, 1)

    # y_hat = torch.round(y_hat).detach().numpy()

    accuracy = np.mean(np.where(y_hat.detach().numpy() == dataset.y_test, 1, 0))

    wandb.run.summary["test_accuracy"] = accuracy

    return accuracy, trtime