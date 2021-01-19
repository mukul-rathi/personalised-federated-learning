from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from glob import glob
import os
import numpy as np 
import matplotlib.pyplot as plt

def set_size(fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    width = 397.48499
    
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim
    
# Using seaborn's style
plt.style.use('seaborn')

# With LaTex fonts
tex_fonts = {
# Use LaTeX to write all text
"text.usetex": False,
"font.family": "serif",
# Use 10pt font in plots, to match 10pt font in document
"axes.labelsize": 10,
"font.size": 10,
# Make the legend/label fonts a little smaller
"legend.fontsize": 8,
"xtick.labelsize": 8,
"ytick.labelsize": 8
}
plt.rcParams.update(tex_fonts)

for directory,_ , _ in os.walk("./completed_runs"):
    clients_dirs = glob(directory + "/client*/")

    

    
    # Initialise figure instance
    test_fig, test_ax = plt.subplots(1, 1, figsize=set_size(0.45))
    train_fig, train_ax = plt.subplots(1, 1, figsize=set_size(0.45))

    # Plot
    test_ax.set_xlim(0, 50)
    test_ax.set_xlabel("Epochs")
    test_ax.set_ylabel("Accuracy (test)")
    
    train_ax.set_xlim(0, 50)
    train_ax.set_xlabel("Epochs")
    train_ax.set_ylabel("Accuracy (train)")
    
    test_accs = []
    train_accs = []

    for client_d in clients_dirs:
        event_acc = EventAccumulator(client_d)
        event_acc.Reload()
        # Show all tags in the log file
        # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'|
        #print(*event_acc.Scalars('Accuracy/test'))
        _, test_step, test_acc = zip(*event_acc.Scalars('Accuracy/test'))
        test_accs.append((test_step,test_acc))
        #_, train_step, train_acc = zip(*event_acc.Scalars('Accuracy/train'))
        #train_accs.append((train_step,train_acc))
        
        
    if test_accs:
        for (x,y) in test_accs:
            test_ax.plot(x, y)
        print(directory)
        test_fig.savefig(directory + '/test_acc.pdf', format='pdf', bbox_inches='tight')
    if train_accs:
        for (x,y) in train_accs:
            train_ax.plot(x, y)
        train_fig.savefig(directory + '/train_acc.pdf', format='pdf', bbox_inches='tight')
    plt.close(train_fig)
    plt.close(test_fig)
