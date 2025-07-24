import math
import numpy as np

import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
from torch.autograd import Variable
from torchvision import transforms


class VAEModel(nn.Module):
    # x --> fc1 --> relu --> fc2 --> z --> fc3 --> relu -->fc4 --> x'
    def __init__(self, input_size, layer1, layer2, device):
        '''
        layer1: dim of hidden layer1
        layer2: dim of hidden layer2
        '''
        super(VAEModel, self).__init__()

        self.input_size = input_size
        self.device = device

        self.fc1 = nn.Linear(input_size, layer1)
        self.fc21 = nn.Linear(layer1, layer2)  # encode
        self.fc22 = nn.Linear(layer1, layer2)  # encode
        self.fc3 = nn.Linear(layer2, layer1)  # decode
        self.fc4 = nn.Linear(layer1, input_size)  # decode

        self.relu = nn.ReLU()
        self.dout = nn.Dropout(p=0.2)

        # initialize weights
        nn.init.xavier_uniform(self.fc1.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.fc21.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.fc22.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.fc3.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.fc4.weight, gain=np.sqrt(2))

    def encode(self, x):
        # x --> fc1 --> relu --> fc21
        # x --> fc1 --> relu --> fc22
        dx = self.dout(x)
        h1 = self.relu(self.fc1(dx))
        h1 = self.dout(h1)
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        eps = Variable(eps)
        return eps * std + mu

    def decode(self, z, x):
        # z --> fc3 --> relu --> fc4
        dz = self.dout(z)
        h3 = self.relu(self.fc3(dz))
        h3 = self.dout(h3)
        return self.fc4(h3).view(x.size())

    def forward(self, x):
        # flatten input and pass to encode
        # mu= mean
        # logvar = log variational
        mu, logvar = self.encode(x.view(-1, self.input_size))
        z = self.reparametrize(mu, logvar)
        return self.decode(z, x), mu, logvar

