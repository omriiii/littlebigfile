import os.path
import random
import sys
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os

class Dataloaderrrr(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
DEVICE = "cpu"
print(f"Script using {DEVICE} device")


class NeuralNetwork(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 layer_size: int = 512,
                 layer_cnt: int = 4):

        super().__init__()
        self.flatten = nn.Flatten()

        layers = [nn.Linear(input_size, layer_size), nn.LeakyReLU()]
        for _ in range(layer_cnt - 1):
            layers.extend([nn.Linear(layer_size, layer_size), nn.LeakyReLU()])

        if output_size == 2:
            layers.extend([nn.Linear(layer_size, output_size), nn.Softmax(dim=-1)])
        elif output_size == 1:
            layers.extend([nn.Linear(layer_size, output_size), nn.Sigmoid()])

        self.linear_relu_stack = nn.Sequential(*layers)
        """
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, 2),
            nn.Softmax(dim=-1),
        )
        """

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

def fourier_series_coeff_numpy(f, T, N, return_complex=False):
    """Calculates the first 2*N+1 Fourier series coeff. of a periodic function.

    Given a periodic, function f(t) with period T, this function returns the
    coefficients a0, {a1,a2,...},{b1,b2,...} such that:

    f(t) ~= a0/2+ sum_{k=1}^{N} ( a_k*cos(2*pi*k*t/T) + b_k*sin(2*pi*k*t/T) )

    If return_complex is set to True, it returns instead the coefficients
    {c0,c1,c2,...}
    such that:

    f(t) ~= sum_{k=-N}^{N} c_k * exp(i*2*pi*k*t/T)

    where we define c_{-n} = complex_conjugate(c_{n})

    Refer to wikipedia for the relation between the real-valued and complex
    valued coeffs at http://en.wikipedia.org/wiki/Fourier_series.

    Parameters
    ----------
    f : the periodic function, a callable like f(t)
    T : the period of the function f, so that f(0)==f(T)
    N_max : the function will return the first N_max + 1 Fourier coeff.

    Returns
    -------
    if return_complex == False, the function returns:

    a0 : float
    a,b : numpy float arrays describing respectively the cosine and sine coeff.

    if return_complex == True, the function returns:

    c : numpy 1-dimensional complex-valued array of size N+1

    """
    # From Shanon theoreom we must use a sampling freq. larger than the maximum
    # frequency you want to catch in the signal.
    f_sample = 2 * N
    # we also need to use an integer sampling frequency, or the
    # points will not be equispaced between 0 and 1. We then add +2 to f_sample
    t, dt = np.linspace(0, T, f_sample + 2, endpoint=False, retstep=True)

    y = np.fft.rfft(f(t)) / t.size

    if return_complex:
        return y
    else:
        y *= 2
        return y[0].real, y[1:-1].real, -y[1:-1].imag

from numpy import ones_like, cos, pi, sin, allclose

'''
def f(t):
    return t

T = 1  # any real number
N_chosen = 3
a0, a, b = fourier_series_coeff_numpy(f, T, N_chosen)

'''


def chunkFileToTrainingData(file_bytearray: bytearray,
                            bits_predicted_per_neural_net_inference: int,
                            fourier_series_terms: int,
                            user_fourier_series_coeffs: bool) -> (np.array, np.array):


    """
    file_bits = [[int(bit) for bit in format(byte, '08b')] for byte in file_bytearray]
    if bits_predicted_per_neural_net_inference == 8:
        pass
    elif bits_predicted_per_neural_net_inference == 1:
        file_bits = [a for b in file_bits for a in b] # flatten
    else:
        raise NotImplementedError(":(")
    y = np.array(file_bits)
    """
    pass
    BYTE = 8
    KILO = 1024
    MEGABYTE = BYTE * KILO * KILO
    GIGABYTE = MEGABYTE * KILO
    y = np.array([[round(random.random())] for i in range(MEGABYTE)], dtype='float32')


    # Create x
    # [0, 1] samples
    X = [[i / (len(y)-1)] for i in range(len(y))]

    if user_fourier_series_coeffs:
        a0, COS_COEFFS, SIN_COEFFS = fourier_series_coeff_numpy(lambda arg:arg, 1, fourier_series_terms)

        for k, coeffs in enumerate(zip(COS_COEFFS, SIN_COEFFS)):
            cos_coeff, sin_coeff = coeffs
            for x in X:
                x.append(cos_coeff*cos(2*pi*(k+1)*x[0]))
                x.append(sin_coeff*sin(2*pi*(k+1)*x[0]))
    else:
        for x in X:
            x.extend([cos(2 * pi * k * x[0]) for k in range(1, fourier_series_terms)])
            x.extend([sin(2 * pi * k * x[0]) for k in range(1, fourier_series_terms)])

    X = np.array(X)



    return X, y

pass


def train(x: np.array,
          y: np.array,
          layer_cnt: int,
          layer_size: int,
          epochs: int,
          learning_rate: float,
          batch_size: int,
          shuffle: bool):


    mlp = NeuralNetwork(input_size=x.shape[1],
                        output_size=y.shape[1],
                        layer_cnt=layer_cnt,
                        layer_size=layer_size)


    optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    dataloader = DataLoader(dataset=Dataloaderrrr(x, y), batch_size=batch_size, shuffle=shuffle)

    mlp.train()
    for epoch in range(epochs):
        correct_labels = torch.tensor(0)

        for batch, (batch_x, batch_y) in enumerate(dataloader):
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            pred = mlp(batch_x)

            loss = loss_fn(pred, batch_y)
            correct_labels += torch.sum(torch.round(pred) == batch_y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        accurecy = correct_labels / len(dataloader.dataset)

        print(f"{epoch}: accurecy: {accurecy}")
        """
        if batch % 1000 == 0:
            loss = loss.item()
            print(f"[{datetime.datetime.fromtimestamp(time.time())}] loss: {loss:>7f}  [{batch:>5d}/{size:>5d}]")
        """

def getFileByteArray(fname: str) -> bytearray:
    with open(fname, 'rb') as f:
        return f.read()


if __name__ == '__main__':

    # USAGE: python3 main.py filename
    fname = sys.argv[1]
    if not os.path.isfile(fname):
        raise RuntimeError(f"Cannot find file {fname}")


    file_bytearray = getFileByteArray(fname)

    x, y = chunkFileToTrainingData(file_bytearray=file_bytearray,
                                   bits_predicted_per_neural_net_inference=1,
                                   fourier_series_terms=1000,
                                   user_fourier_series_coeffs=False
                                   #output_is_categorized=False
                                   )


    train(x=x,
          y=y,
          layer_cnt=4,
          layer_size=512,
          epochs=10000,
          learning_rate=0.00075,
          batch_size=4096,
          shuffle=True)

