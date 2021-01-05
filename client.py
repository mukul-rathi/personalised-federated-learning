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
"""Flower client example using PyTorch for CIFAR-10 image classification."""


import argparse
import timeit
from collections import OrderedDict
from typing import Optional
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision

import flwr as fl
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes, Weights

import models.cifar as cifar

DEFAULT_SERVER_ADDRESS = "[::]:8080"

DEVICE = torch.device("cpu")

class CifarClient(fl.client.Client):
    """Flower client implementing CIFAR-10 image classification using PyTorch."""

    def __init__(
        self,
        cid: str,
        model: cifar.Net,
        trainset: torchvision.datasets.CIFAR10,
        testset: torchvision.datasets.CIFAR10,
        exp_name: Optional[str],
        iid_fraction: Optional[float]
    ) -> None:
        self.cid = cid
        self.model = model
        self.trainset = trainset
        self.testset = testset
        if exp_name:
            self.exp_name = exp_name
        else:
            exp_name = 'federated_unspecified'

    def get_parameters(self) -> ParametersRes:
        print(f"Client {self.cid}: get_parameters")

        weights: Weights = self.model.get_weights()
        parameters = fl.common.weights_to_parameters(weights)
        return ParametersRes(parameters=parameters)

    def fit(self, ins: FitIns) -> FitRes:
        print(f"Client {self.cid}: fit")

        weights: Weights = fl.common.parameters_to_weights(ins.parameters)
        
        config = ins.config
        
        fit_begin = timeit.default_timer()

        # Get training config
        epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])
        epoch_global = int(config["epoch_global"])
        
        # Generate Client experiment label
        #d = OrderedDict(sorted(config.items()))
        #params = '_'.join([f'{k}_{v}' for k,v in d.items()])
        client_name = f'client_{self.cid}_{self.exp_name}'

        # Set model parameters
        self.model.set_weights(weights)

        # Train model
        trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=batch_size, shuffle=True
        )
        
        start_epoch = epoch_global+1
        end_epoch = start_epoch + epochs-1
        results_fit = cifar.train(net=self.model, trainloader = trainloader, 
                                            device = DEVICE, start_epoch=start_epoch, end_epoch = end_epoch)
        # Write to tensorboard 
        with SummaryWriter(log_dir=f'./runs/{client_name}') as writer:
            for idx, result in enumerate(results_fit, start_epoch):
                loss, acc = result
                writer.add_scalar('Loss/train', loss, idx)
                writer.add_scalar('Accuracy/train', acc, idx)

        # Return the refined weights and the number of examples used for training
        weights_prime: Weights = self.model.get_weights()
        params_prime = fl.common.weights_to_parameters(weights_prime)
        num_examples_train = len(self.trainset)
        fit_duration = timeit.default_timer() - fit_begin
        return FitRes(
            parameters=params_prime,
            num_examples=num_examples_train,
            num_examples_ceil=num_examples_train,
            fit_duration=fit_duration,
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print(f"Client {self.cid}: evaluate")
        config = ins.config
        epoch_global = int(config["epoch_global"])
        
        # Generate Client experiment label
        d = OrderedDict(sorted(config.items()))
        params = '_'.join([f'{k}_{v}' for k,v in d.items()])
        exp_name = f'client_{self.cid}' + params
        
        weights = fl.common.parameters_to_weights(ins.parameters)

        # Use provided weights to update the local model
        self.model.set_weights(weights)

        # Evaluate the updated model on the local dataset
        testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=32, shuffle=False
        )
        
        loss, accuracy = test(net=self.model, testloader=testloader, device = device, epoch_global=epoch_global, exp_name=exp_name)

        # Return the number of evaluation examples and the evaluation result (loss)
        return EvaluateRes(
            num_examples=len(self.testset), loss=float(loss), accuracy=float(accuracy)
        )


def main() -> None:
    """Load data, create and start CifarClient."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--server_address",
        type=str,
        default=DEFAULT_SERVER_ADDRESS,
        help=f"gRPC server address (default: {DEFAULT_SERVER_ADDRESS})",
    )
    parser.add_argument(
        "--cid", type=str, required=True, help="Client CID (no default)"
    )
    parser.add_argument(
        "--num_partitions", 
        type=int, 
        required=True, 
        help="In our case, this is the total number of clients participating during training. The original dataset is partitioned among clients."
    )
    parser.add_argument(
        "--iid_fraction", 
        type=float, 
        nargs="?", 
        const=1.0, 
        help="Fraction of data [0,1] that is independent and identically distributed."
    )
    parser.add_argument(
        "--log_host",
        type=str,
        help="Logserver address (no default).",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        help="Useful experiment name for tensorboard plotting.",
    )
    args = parser.parse_args()

    # Configure logger
    fl.common.logger.configure(f"client_{args.cid}", host=args.log_host)

    # Load model and data
    model = cifar.load_model()
    model.to(DEVICE)
    trainset, testset = cifar.load_data()
    print(f'Loading data for client {args.cid}')
    trainset, testset = cifar.load_local_partitioned_data(cid=int(args.cid), 
                                                          iid_fraction = args.iid_fraction, 
                                                          num_partitions = args.num_partitions)
    # Start client
    print(f'Starting client {args.cid}')
    client = CifarClient(args.cid, model, trainset, testset, f'{args.exp_name}_iid-fraction_{args.iid_fraction}', args.iid_fraction)
    fl.client.start_client(args.server_address, client)


if __name__ == "__main__":
    main()

