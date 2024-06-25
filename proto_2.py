import os.path
import random
import sys
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os
import math

class FileBytearrayAsDataset(Dataset):
    def __init__(self,
                 file_bytearray: bytearray,
                 fourier_series_terms: int,
                 lazy_eval:bool=True):
        self.file_bytearray = file_bytearray
        self.fourier_series_terms = fourier_series_terms
        self.samples_cnt = len(self.file_bytearray)

        #
        # Create x

        # [0, 1] samples as floats
        # eg. 6 sample_cnt would yield array of [0, 0.2, 0.4, 0.6, 0.8, 1]
        self.x = [i / (self.samples_cnt - 1) for i in range(self.samples_cnt)]

        """
        # Append fourier terms
        if not lazy_eval:
            for k in range(fourier_series_terms):
                for x in X:
                    x.append(cos(2 * pi * (k + 1) * x[0]))
                    x.append(sin(2 * pi * (k + 1) * x[0]))


        self.x = torch.from_numpy(np.array(, dtype='float32'))
        """

        #self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):

        #
        # CALCULATE X

        # Our intermediate x
        # Remember, x here is just some float from 0 to 1
        x = self.x[idx]

        ret_x = [x]

        for k in range(self.fourier_series_terms):
            ret_x.append(cos(2 * pi * (k + 1) * x))
            ret_x.append(sin(2 * pi * (k + 1) * x))

        ret_x = torch.from_numpy(np.array(ret_x, dtype='float32'))

        #
        #   CALCULATE Y
        byte_index = math.floor(idx/8)
        byte = self.file_bytearray[byte_index]
        bit_location = idx%8
        bit = (byte << bit_location) & 1

        ret_y = torch.from_numpy(np.array([bit], dtype='float32'))

        return ret_x, ret_y

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

    y = np.array([[round(random.random())]    for i in range(1024*16)])


    # Create x
    # [0, 1] samples
    X = [[i / (len(y)-1)] for i in range(len(y))]

    if user_fourier_series_coeffs:
        a0, COS_COEFFS, SIN_COEFFS = fourier_series_coeff_numpy(lambda arg:arg, 1, fourier_series_terms)
    else:
        COS_COEFFS = [1]*fourier_series_terms
        SIN_COEFFS = [1]*fourier_series_terms

    for k, coeffs in enumerate(zip(COS_COEFFS, SIN_COEFFS)):
        cos_coeff, sin_coeff = coeffs
        for x in X:
            x.append(cos_coeff*cos(2*pi*(k+1)*x[0]))
            x.append(sin_coeff*sin(2*pi*(k+1)*x[0]))

    X = np.array(X)



    return X, y

pass


def train(file_bytearray: bytearray,
          fourier_series_terms: int,
          layer_cnt: int,
          layer_size: int,
          epochs: int,
          learning_rate: float,
          batch_size: int,
          shuffle: bool):

    dataset = FileBytearrayAsDataset(file_bytearray=file_bytearray,
                                     fourier_series_terms=fourier_series_terms)
    print(f"Training on {len(dataset)} samples")
    print(f"Batch size {batch_size}")

    temp_sample = dataset[0]

    print("Creating data loader...")
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle)

    print("Creating MLP...")
    mlp = NeuralNetwork(input_size=temp_sample[0].shape[0],
                        output_size=1,
                        layer_cnt=layer_cnt,
                        layer_size=layer_size)



    optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()


    mlp.train()
    for epoch in range(epochs):
        correct_labels = torch.tensor(0)

        for batch, (batch_x, batch_y) in enumerate(dataloader):
            print(f'{batch} / {math.ceil(len(dataloader.dataset) / batch_size)}')
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

    torch.save(mlp, "test_mlp")

def getFileByteArray(fname: str) -> bytearray:
    with open(fname, 'rb') as f:
        return f.read()


if __name__ == '__main__':

    # USAGE: python3 main.py filename
    fname = sys.argv[1]
    if not os.path.isfile(fname):
        raise RuntimeError(f"Cannot find file {fname}")

    # lol. override fname
    fname = 'some_file.asdf'
    file_bytearray = getFileByteArray(fname)


    train(file_bytearray=file_bytearray,
          fourier_series_terms=1000,
          #bits_predicted_per_neural_net_inference=1,
          #bits_predicted_per_neural_net_inference=1,
          layer_cnt=4,
          layer_size=512,
          epochs=250,
          learning_rate=0.00075,
          batch_size=4096*8,
          shuffle=False)

