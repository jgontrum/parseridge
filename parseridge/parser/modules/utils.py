from torch import nn


def init_weights_xavier(network, activation="tanh"):
    """
    Initializes the layers of a given network with random values.
    Bias layers will be filled with zeros.

    Parameters
    ----------
    network : torch.nn object
        The network to initialize.
    """
    for name, param in network.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            nn.init.xavier_normal_(
                param, gain=nn.init.calculate_gain(activation)
            )
