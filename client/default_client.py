from collections import OrderedDict
from typing import Optional
import timeit
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision
from models import cifar

import flwr as fl
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes, Weights

DEVICE = torch.device("cpu")


class DefaultClient(fl.client.Client):
    """Default Flower client using global model's weights."""

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
        self.exp_name = exp_name if exp_name else 'federated_unspecified'


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
        results_fit = self.model.train(trainloader = trainloader, 
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
        
        loss, accuracy = test(net=self.model, testloader=testloader, device = DEVICE, epoch_global=epoch_global, exp_name=exp_name)

        # Return the number of evaluation examples and the evaluation result (loss)
        return EvaluateRes(
            num_examples=len(self.testset), loss=float(loss), accuracy=float(accuracy)
        )
