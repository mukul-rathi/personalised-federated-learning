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
"""Minimal example on how to start a simple Flower server."""


import argparse
from typing import Callable, Dict, Optional, Tuple
from collections import OrderedDict

from logging import INFO
from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.common.logger import log
from flwr.server.grpc_server.grpc_server import start_insecure_grpc_server

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

import flwr as fl
import models.cifar as cifar
  
DEFAULT_SERVER_ADDRESS = "[::]:8080"

DEVICE = torch.device("cpu")

def main() -> None:
    """Start server and train five rounds."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--server_address",
        type=str,
        default=DEFAULT_SERVER_ADDRESS,
        help=f"gRPC server address (default: {DEFAULT_SERVER_ADDRESS})",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of rounds of federated learning (default: 1)",
    )
    parser.add_argument(
        "--sample_fraction",
        type=float,
        default=1.0,
        help="Fraction of available clients used for fit/evaluate (default: 1.0)",
    )
    parser.add_argument(
        "--min_sample_size",
        type=int,
        default=2,
        help="Minimum number of clients used for fit/evaluate (default: 2)",
    )
    parser.add_argument(
        "--min_num_clients",
        type=int,
        default=2,
        help="Minimum number of available clients required for sampling (default: 2)",
    )
    parser.add_argument(
        "--log_host",
        type=str,
        help="Logserver address (no default)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs each client will train for (default: 10)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size used when training each client (default: 32)",
    )   
    parser.add_argument(
        "--exp_name",
        type=str,
        help="Name of the experiment you are running (no default)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="fedAvg",
        help="Name of the strategy you are running (default: fedAvg)",
    )
    
    parser.add_argument(
        "--q_param",
        type=float,
        help="Q param for QFFedAvg",
    )
    
    parser.add_argument(
        "--qffl_learning_rate",
        type=float,
        help="Learning_rate for QFFedAvg",
    )
    
    args = parser.parse_args()
    dict_args = vars(args)
    if not args.exp_name:
        tags = ['strategy', 'rounds', 'epochs', 'min_num_clients', 'min_sample_size', 'sample_fraction']
        params = '_'.join([f'{tag}_{dict_args[tag]}' for tag in tags])
        args.exp_name=f'federated_' + params

    # Configure logger
    fl.common.logger.configure("server", host=args.log_host)

    # Load evaluation data
    _, testset = cifar.load_data()
    

    # Create client_manager, strategy, and server
    client_manager = fl.server.SimpleClientManager()
    strategy = get_strategy(args, testset);
    server = fl.server.Server(client_manager=client_manager, strategy=strategy)
    
    # Run server 
    grpc_server = start_insecure_grpc_server(
        client_manager=server.client_manager(),
        server_address=DEFAULT_SERVER_ADDRESS,
        max_message_length=GRPC_MAX_MESSAGE_LENGTH,
    )
    
    # Fit model
    hist = server.fit(num_rounds=args.rounds)
    log(INFO, "app_fit: losses_distributed %s", str(hist.losses_distributed))
    log(INFO, "app_fit: accuracies_distributed %s", str(hist.accuracies_distributed))
    log(INFO, "app_fit: losses_centralized %s", str(hist.losses_centralized))
    log(INFO, "app_fit: accuracies_centralized %s", str(hist.accuracies_centralized))
    print(hist.accuracies_centralized)
    
    with SummaryWriter(log_dir=f'./runs/{args.exp_name}') as writer:
        for idx, loss in hist.losses_centralized:
            writer.add_scalar('Loss/test', loss, idx*args.epochs)
        for idx, acc in hist.accuracies_centralized:
            writer.add_scalar('Accuracy/test', acc, idx*args.epochs)

    # Temporary workaround to force distributed evaluation
    server.strategy.eval_fn = None  # type: ignore

    # Evaluate the final trained model
    res = server.evaluate(rnd=-1)
    if res is not None:
        loss, (results, failures) = res
        log(INFO, "app_evaluate: federated loss: %s", str(loss))
        log(
            INFO,
            "app_evaluate: results %s",
            str([(res[0].cid, res[1]) for res in results]),
        )
        log(INFO, "app_evaluate: failures %s", str(failures))
    else:
        log(INFO, "app_evaluate: no evaluation result")

    # Stop the gRPC server
    grpc_server.stop(1)
    

def generate_config(args):  
    """Returns a funtion of parameters based on arguments"""
    
    def fit_config(rnd: int) -> Dict[str, str]:
        config = {
        "epoch_global": str((rnd-1)*args.epochs),
        "epochs": str(args.epochs),
        "batch_size": str(args.batch_size),
        }
        return config
    
    return fit_config 


def get_eval_fn(
    testset: torchvision.datasets.CIFAR10,
) -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""
    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-10 test set for evaluation."""
        model = cifar.load_model()
        model.set_weights(weights)
        model.to(DEVICE)
        testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
        return cifar.test(net=model, testloader=testloader, device = DEVICE)

    return evaluate


def get_strategy(
    args,
    testset: torchvision.datasets.CIFAR10
) -> fl.server.strategy.Strategy:
    if args.strategy == "fedAvg":
        return fl.server.strategy.FedAvg(
        fraction_fit=args.sample_fraction,
        fraction_eval=args.sample_fraction,
        min_fit_clients=args.min_sample_size,
        min_eval_clients=args.min_sample_size,
        min_available_clients=args.min_num_clients,
        eval_fn= get_eval_fn(testset),
        on_fit_config_fn=generate_config(args),
        on_evaluate_config_fn=generate_config(args)
        )
    elif args.strategy == "qffedAvg":
        if not args.q_param:
            args.q_param = 0.2
        if not args.qffl_learning_rate:
            args.qffl_learning_rate = 0.1
        
        return fl.server.strategy.QffedAvg(
        q_param = args.q_param,
        qffl_learning_rate = args.qffl_learning_rate,
        fraction_fit=args.sample_fraction,   
        fraction_eval=args.sample_fraction,
        min_fit_clients=args.min_sample_size,
        min_eval_clients=args.min_sample_size,
        min_available_clients=args.min_num_clients,
        eval_fn= get_eval_fn(testset),
        on_fit_config_fn=generate_config(args),
        on_evaluate_config_fn=generate_config(args)
        )



if __name__ == "__main__":
    main()

