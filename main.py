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


def chunkFileToTrainingData(file_bytearray: bytearray,
                            chunk_size: int) -> (np.array, np.array):
    # debug!
    fake_debug_length = 10000
    x = np.array([[i / (fake_debug_length-1)] for i in range(fake_debug_length)])
    y = np.array([[round(random.random())]    for i in range(fake_debug_length)])

    # unfurl X to a fast foruier transform
    return x, y


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
                                   chunk_size=8)

    train(x=x,
          y=y,
          layer_cnt=6,
          layer_size=512,
          epochs=1000,
          learning_rate=0.00075,
          batch_size=64,
          shuffle=True)

