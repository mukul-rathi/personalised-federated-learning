from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from glob import glob
import os
import numpy as np 

for directory,_ , _ in os.walk("./completed_runs"):
    clients_dirs = glob(directory + "/client*/")


    client_accuracies = []

    for client_d in clients_dirs:
        event_acc = EventAccumulator(client_d)
        event_acc.Reload()
        # Show all tags in the log file
        # print(event_acc.Tags())
        # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'|
        _, _, test_acc = zip(*event_acc.Scalars('Accuracy/test'))
        client_accuracies.append(test_acc[-1])

    if client_accuracies:
        print("")
        print(directory)
        mean = np.mean(client_accuracies)
        std = np.std(client_accuracies)
        x = np.array(client_accuracies)
        idx = (x).argsort()[-2:]
        top2 = np.mean(x[idx])
        idx = (x).argsort()[:2]
        bottom2 = np.mean(x[idx])
 
        print(f'{mean:.3f} & {std:.3f} & {top2 :.3f} & {bottom2 :.3f}')