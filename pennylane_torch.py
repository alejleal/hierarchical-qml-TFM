import numpy as np
import torch
from torch import nn

from architecture import get_circuit, get_qcnn, a, b, g, poolg
import time

device = 'default.qubit.torch'
interface = 'torch'
qcnn = get_qcnn(g, poolg)

circuit = get_circuit(qcnn, device, interface)

# set up train loop
def train(x, y, circuit, symbols, epochs, lr, verbose=True):
    opt = torch.optim.Adam([symbols], lr)
    loss = nn.BCELoss()

    for it in range(epochs):
        opt.zero_grad()

        y_hat = circuit(x, symbols)
        loss_eval = loss(y_hat, y)    # y_hat[:, 1]
        loss_eval.backward()

        opt.step()

        if verbose:
            if it % 5 == 0:
                print(f"Loss at step {it}: {loss_eval}")

    return symbols, loss

def run_torch(dataset, epochs, lr, verbose=False):
    # parameter initialization
    n_symbols = qcnn.n_symbols
    symbols = torch.rand(n_symbols, requires_grad=True)

    # build the circuit
    # circuit = get_circuit(qcnn, device="default.qubit")
    y_train_tensor = torch.tensor(dataset.y_train, dtype=torch.double)

    # train qcnn
    t0 = time.time()
    symbols, loss = train(dataset.x_train, y_train_tensor, circuit, symbols, epochs, lr, verbose=verbose)
    trtime = time.time() - t0

    # get predictions
    y_hat = circuit(dataset.x_test, symbols)

    # evaluate
    # y_hat = torch.argmax(y_hat, axis=1).detach().numpy()
    y_hat = torch.round(y_hat).detach().numpy()

    accuracy = np.mean(np.where(y_hat == dataset.y_test, 1, 0))

    return accuracy, symbols, trtime

## Testing grounds
if __name__ == "__main__":
    # test_params()
    print(torch.cuda.is_available())
    try:
        print(torch.cuda.get_device_name(0))
    except:
        pass