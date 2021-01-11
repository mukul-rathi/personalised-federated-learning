# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""PyTorch CIFAR-10 image classification.

The code is generally adapted from 'PyTorch: A 60 Minute Blitz'. Further
explanations are given in the official PyTorch tutorial:

https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""

from collections import OrderedDict
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch import Tensor
from tqdm import tqdm

from flwr_experimental.baseline.dataset.dataset import (
    XY,
    PartitionedDataset,
    create_partitioned_dataset,
    log_distribution,
)

import flwr as fl

DATA_ROOT = "~/.flower/data/cifar-10"

class Net(nn.Module):
    """Simple CNN adapted from 'PyTorch: A 60 Minute Blitz'."""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: Tensor) -> Tensor:
        """Compute forward pass."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_weights(self) -> fl.common.Weights:
        """Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_weights(self, weights: fl.common.Weights) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        state_dict = OrderedDict(
            {k: torch.Tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )
        self.load_state_dict(state_dict, strict=True)
    
    def train(
        self,
        trainloader: torch.utils.data.DataLoader,
        device: torch.device,
        start_epoch: int,
        end_epoch: int
    ) -> List[Tuple[float, float]]:
        """Train the network."""
        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

        print(f"Training from epoch(s) {start_epoch} to {end_epoch} w/ {len(trainloader)} batches each.", flush=True)
        results = []
        # Train the network
        for idx, epoch in enumerate(range(start_epoch, end_epoch+1)):  # loop over the dataset multiple times, last epoch inclusive
            running_loss = 0.0
            running_acc  = 0.0
            total = 0
            pbar = tqdm(trainloader, 0)
            for data in pbar:
                pbar.set_description(f'Epoch {epoch}: Training...')
                images, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # collect statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1) 
                total += labels.size(0)
                running_acc += (predicted == labels).sum().item()

            results.append((running_loss/total, running_acc/total))    

        return results      

    def test(
        self,
        testloader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> Tuple[float, float]:
        """Validate the network on the entire test set."""
        criterion = nn.CrossEntropyLoss()
        correct = 0.0
        total = 0
        loss = 0.0
        with torch.no_grad():
            pbar = tqdm(testloader)
            for idx, data in enumerate(pbar):
                pbar.set_description(f'Testing...')
                images, labels = data[0].to(device), data[1].to(device)
                outputs = self(images)
                loss += criterion(outputs, labels).item() #Average?
                _, predicted = torch.max(outputs.data, 1) 
                total += labels.size(0)
                correct += (predicted == labels).sum().item()  

        return (loss/total, correct/total)




def load_model() -> Net:
    """Load a simple CNN."""
    return Net()


def load_data() -> Tuple[torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR10]:
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=False, download=True, transform=transform
    )
    return trainset, testset

class PartitionedDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, X:torch.Tensor, Y, transform=None):
        self.X = X
        self.Y = Y
    
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (self.X[idx], int(self.Y[idx]))

def load_local_partitioned_data(cid:int, iid_fraction: float, num_partitions: int) -> Tuple[torch.utils.data.Dataset,torch.utils.data.Dataset]:
    """Load partitioned version of CIFAR-10."""
    
    trainset, testset = load_data()
    
    train_loader = DataLoader(trainset, batch_size=len(trainset))
    test_loader = DataLoader(testset, batch_size=len(testset))

    xy_train = ( next(iter(train_loader))[0].numpy(), next(iter(train_loader))[1].numpy() )
    xy_test = ( next(iter(test_loader))[0].numpy(), next(iter(test_loader))[1].numpy() )
    
    xy_train_test_partitions, xy_test = create_partitioned_dataset(
        (xy_train, xy_test), iid_fraction, num_partitions
    )
 
    xy_train_partitions, xy_test_partitions = xy_train_test_partitions 
    
    this_train_data = xy_train_partitions[cid]
    x_train, y_train = this_train_data
    torch_partition_trainset = PartitionedDataset(torch.Tensor(x_train), y_train )
    #torch_partition_trainset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
    
    this_test_data = xy_test_partitions[cid]
    x_test, y_test = this_test_data
    #torch_partition_testset = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
    torch_partition_testset = PartitionedDataset(torch.Tensor(x_test), y_test )

    
    return torch_partition_trainset, torch_partition_testset



