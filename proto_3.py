import datetime
import os.path
import random
import sys
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import time
import os
import math

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Script using {DEVICE} device")

# Print iterations progress
def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = '')
    # Print New Line on Complete
    if iteration == total:
        print()


class FileBytearrayAsDataset(Dataset):
    def __init__(self,
                 file_bytearray: bytearray,
                 fourier_series_terms: int,
                 lazy_eval:bool=True):
        self.file_bytearray = file_bytearray
        self.fourier_series_terms = fourier_series_terms
        self.samples_cnt = len(self.file_bytearray)


        """

        #
        # Create x

        # [0, 1] samples as floats
        # eg. 6 sample_cnt would yield array of [0, 0.2, 0.4, 0.6, 0.8, 1]
        self.x = [i / (self.samples_cnt - 1) for i in range(self.samples_cnt)]

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
        return self.samples_cnt

    def __getitem__(self, idx):

        #
        # CALCULATE X

        # Our intermediate x
        # Remember, x here is just some float from 0 to 1
        #x = idx/(self.samples_cnt-1)

        """
        for k in range(self.fourier_series_terms):
            ret_x.append(cos(2 * pi * (k + 1) * x))
            ret_x.append(sin(2 * pi * (k + 1) * x))

        ret_x = torch.from_numpy(np.array(ret_x, dtype='float32'))
        """
        # Backwards fourier...
        use_absolutes = True

        # Create a tensor with indices
        k = torch.arange(2, self.fourier_series_terms + 2, device=DEVICE)
        #k = torch.arange((self.samples_cnt-1), (self.samples_cnt-1)-self.fourier_series_terms, step=-1, device=DEVICE)/(self.samples_cnt-1)
        angles = 2 * np.pi * (1/k) * idx

        cos_values = torch.cos(angles)
        sin_values = torch.sin(angles)
        """
        if use_absolutes:  
            cos_values = (torch.cos(angles)             / 2) + 0.5
            sin_values = (torch.sin(angles - (np.pi/2)) / 2) + 0.5
        else:
            cos_values = torch.cos(angles)
            sin_values = torch.sin(angles)
        """

        # Interleave cosine and sine values
        ret_x = torch.empty(2 * self.fourier_series_terms, device=DEVICE)
        ret_x[0::2] = cos_values
        ret_x[1::2] = sin_values

        ###########33

        #
        #   CALCULATE Y
        byte_index = math.floor(idx/8)
        byte = self.file_bytearray[byte_index]
        bit_location = idx%8
        bit = (byte >> bit_location) & 1

        ret_y = torch.from_numpy(np.array([bit], dtype='float32'))

        return ret_x, ret_y


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

from numpy import ones_like, cos, pi, sin, allclose

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
                        layer_size=layer_size).to(DEVICE)



    optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    mlp.train()
    for epoch in range(epochs):
        start_timestamp = time.time()
        correct_labels = torch.tensor(0).to(DEVICE)

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

            time_elapsed = time.time() - start_timestamp
            seconds_per_batch = time_elapsed / max(1, batch)
            remaining_batches = math.ceil(len(dataset) / batch_size) - batch
            printProgressBar(iteration=batch,
                             total=math.ceil(len(dataset) / batch_size),
                             length=20,
                             suffix=f" | Time left: {datetime.timedelta(seconds=seconds_per_batch*remaining_batches)}")


        accurecy = correct_labels / len(dataset)

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
    """
    fname = sys.argv[1]
    if not os.path.isfile(fname):
        raise RuntimeError(f"Cannot find file {fname}")
    """

    # lol. override fname
    #fname = '/home/omri/Downloads/Concept Art.zip' # 100mb
    fname = '/home/omri/Downloads/02 Justice - Sure You Will.wav' # 50mb
    #fname = '/home/omri/Downloads/JUSTICE_RIPOFF_4.mpga' # 1.8mb
    file_bytearray = getFileByteArray(fname)


    train(file_bytearray=file_bytearray,
          fourier_series_terms=500,
          #bits_predicted_per_neural_net_inference=1,
          layer_cnt=3,
          layer_size=1024,
          epochs=500,
          learning_rate=0.00075,
          batch_size=512, # 8192
          shuffle=False)

    """
    Compression Efficeny = Original size of file / (NN file size + 'missed bits table')
    """

