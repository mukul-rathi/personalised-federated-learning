from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple, List
import itertools as it
import timeit
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torchvision
from models import fashionmnist as dataset
from tqdm import tqdm
import copy

import flwr as fl
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes, Weights

DEVICE = torch.device("cpu")

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)

class PerFedAvgClient(fl.client.Client):
    """PerFedAvgClient Flower client using a personalised local model's weights."""

    def __init__(
        self,
        cid: str,
        model: dataset.Net,
        trainset: torchvision.datasets.VisionDataset,
        testset: torchvision.datasets.VisionDataset,
        exp_name: Optional[str],
        iid_fraction: Optional[float],
        alpha: Optional[float],
        beta: Optional[float]
    ) -> None:
        self.cid = cid
        self.model = model
        self.trainset = trainset
        self.testset = testset
        self.exp_name = exp_name if exp_name else 'federated_unspecified'
        self.alpha = alpha if alpha else 1e-2
        self.beta = beta if beta else 1e-3

    
    def get_parameters(self) -> ParametersRes:
        print(f"Client {self.cid}: get_parameters")

        weights: Weights = self.model.get_weights()
        parameters = fl.common.weights_to_parameters(weights)
        return ParametersRes(parameters=parameters)

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
        alpha_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.alpha)
        beta_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.beta)

        print(f"Training from epoch(s) {start_epoch} to {end_epoch} w/ {len(trainloader)} batches each.", flush=True)
        results = []
        # Train the network
        for idx, epoch in enumerate(range(start_epoch, end_epoch+1)):  # loop over the dataset multiple times, last epoch inclusive
            running_loss = 0.0
            running_acc  = 0.0
            total = 0
            pbar = tqdm(trainloader, 0)
            for (data1, data2) in pairwise(pbar):
                pbar.set_description(f'Epoch {epoch}: Training...')
                images1, labels1 = data1[0].to(device), data1[1].to(device)
                images2, labels2 = data2[0].to(device), data2[1].to(device)

                # zero the parameter gradients
                alpha_optimizer.zero_grad()

                # forward + backward + optimize
                                
                # copy w_t
                w_t_copy = copy.deepcopy(self.model.get_weights())

                outputs1 = self.model(images1)
                loss = criterion(outputs1, labels1)
                # delta = grad(w_t)
                loss.backward()
                
                # ~w_t = w_t - alpha*delta
                alpha_optimizer.step()
                
                beta_optimizer.zero_grad()
                # f(~w_t)
                outputs2 = self.model(images2)
                loss = criterion(outputs2, labels2)
                
                # compute ~delta = grad of(~w_t)
                loss.backward()
                
                self.model.set_weights(w_t_copy)
                # w_t+1 =  w_t - beta * ~delta
                beta_optimizer.step()

                # collect statistics
                running_loss += loss.item() # get loss of f(w - alpha*grad)
                _, predicted2 = torch.max(outputs2.data, 1) 
                total += labels1.size(0)
                running_acc += (predicted2 == labels2).sum().item()

            results.append((running_loss/total, running_acc/total))    

        return results 
    
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
        results_fit = self.train(trainloader = trainloader, 
                                        device = DEVICE, 
                                       start_epoch=start_epoch, end_epoch = end_epoch)
        
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
       # d = OrderedDict(sorted(config.items()))
       # params = '_'.join([f'{k}_{v}' for k,v in d.items()])
        #exp_name = f'client_{self.cid}' + params
        
        client_name = f'client_{self.cid}_{self.exp_name}'
        
        weights = fl.common.parameters_to_weights(ins.parameters)
        
        weights_copy = copy.deepcopy(weights)
        
        # Use provided weights to update the local model
        self.model.set_weights(weights)
        
        # Get test dataset
        testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=32, shuffle=True
        )
        
        # Take one step (personalise model)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.alpha)
        optimizer.zero_grad()
        # forward + backward + optimize
        data = next(iter(testloader))
        images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
        outputs = self.model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Evaluate the updated model on the local dataset
        loss, accuracy =self.model.test(testloader=testloader, device = DEVICE)
        
        # Write to tensorboard 
        with SummaryWriter(log_dir=f'./runs/{client_name}') as writer:
                writer.add_scalar('Loss/test', loss, epoch_global)
                writer.add_scalar('Accuracy/test', accuracy, epoch_global)
        
        # We use personalisation for evaluation, but shouldn't affect subsequent training, so set
        # to original params before personalisation.
        self.model.set_weights(weights_copy)

        
        # Return the number of evaluation examples and the evaluation result (loss)
        return EvaluateRes(
            num_examples=len(self.testset), loss=float(loss), accuracy=float(accuracy)
        )
