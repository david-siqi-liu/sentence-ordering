import random
import numpy as np
import torch


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Current device: {:s}".format(device.type))
    return device


def set_seed():
    seed = args['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def plot_statistics(measure, values):
    fig = plt.figure(figsize=(20, 10))
    plt.title("train/validation {}".format(measure))
    plt.plot(values['train'], label='train')
    plt.plot(values['val'], label='val')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel(measure)
    plt.legend(loc='best')
