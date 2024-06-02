import torch
import torch.nn as nn
from torch.nn import Conv2d
from torch.nn.utils import prune


class Net(nn.Module):
    def __init__(self,
                 in_channels,
                 number_of_convolutional_layers,
                 out_channels,
                 kernel_size,
                 kernel_stride,
                 pool_size,
                 pool_type,
                 number_of_hidden_layers,
                 number_of_neurons_in_layers,
                 output_size,
                 batch_norm=False,
                 dropout=[],
                 layer_norm=False,
                 weight_norm=False):
        super(Net, self).__init__()
        cnn_layers = []
        dropout_index = 0
        for counter in range(number_of_convolutional_layers):
            if counter == 0:
                conv_layer = Conv2d(in_channels=in_channels, out_channels=out_channels[counter],
                                    kernel_size=kernel_size[counter], stride=kernel_stride[counter], padding=0,
                                    bias=False)
            else:
                conv_layer = Conv2d(in_channels=out_channels[counter - 1], out_channels=out_channels[counter],
                                    kernel_size=kernel_size[counter], stride=kernel_stride[counter], padding=0,
                                    bias=False)
            if weight_norm:
                conv_layer = torch.nn.utils.parametrizations.weight_norm(conv_layer)

            cnn_layers.append(conv_layer)
            if pool_type[counter] == 0:
                cnn_layers.append(nn.AdaptiveMaxPool2d((pool_size[counter],pool_size[counter])))
            else:
                cnn_layers.append(nn.AdaptiveAvgPool2d((pool_size[counter],pool_size[counter])))

            if batch_norm:
                cnn_layers.append(nn.BatchNorm2d(out_channels[counter]))

            cnn_layers.append(nn.LeakyReLU(inplace=True))

            if layer_norm:
                cnn_layers.append(nn.LayerNorm([out_channels[counter], pool_size[counter][0], pool_size[counter][1]]))

            if len(dropout) > 0 and dropout[dropout_index] > 0:
                cnn_layers.append(nn.Dropout2d(p=dropout[dropout_index]))
            dropout_index += 1

        self.cnn_layers = nn.Sequential(*cnn_layers)

        input_size = (out_channels[number_of_convolutional_layers - 1] * pool_size[number_of_convolutional_layers - 1] * pool_size[number_of_convolutional_layers - 1])
        layers = []
        input_layer = nn.Linear(in_features=input_size, out_features=number_of_neurons_in_layers[0])
        if weight_norm:
            layers.append(torch.nn.utils.parametrizations.weight_norm(input_layer))
        else:
            layers.append(input_layer)

        layers.append(nn.LeakyReLU())

        if batch_norm:
            layers.append(nn.BatchNorm1d(number_of_neurons_in_layers[0]))

        if layer_norm:
            layers.append(nn.LayerNorm(number_of_neurons_in_layers[0]))

        if len(dropout) > 0 and dropout[dropout_index] > 0:
            cnn_layers.append(nn.Dropout2d(p=dropout[dropout_index]))
        dropout_index += 1

        if number_of_hidden_layers > 1:
            for counter in range(number_of_hidden_layers - 1):
                layer = nn.Linear(number_of_neurons_in_layers[counter], number_of_neurons_in_layers[counter + 1])
                if weight_norm:
                    layer = torch.nn.utils.parametrizations.weight_norm(layer)
                layers.append(layer)

                if batch_norm:
                    layers.append(nn.BatchNorm1d(number_of_neurons_in_layers[counter + 1]))  # Adjusted batch_norm size

                layers.append(nn.LeakyReLU())

                if layer_norm:
                    layers.append(nn.LayerNorm(number_of_neurons_in_layers[counter + 1]))

                if len(dropout) > 0 and dropout[dropout_index] > 0:
                    cnn_layers.append(nn.Dropout2d(p=dropout[dropout_index]))
                dropout_index += 1
        # Add the output layer
        layers.append(nn.Linear(number_of_neurons_in_layers[number_of_hidden_layers - 1], output_size))
        layers.append(nn.Softmax(dim=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, out):
        for layer in self.cnn_layers:
            out = layer(out)
        out = out.view(out.size(0), -1)
        for layer in self.layers:
            out = layer(out)

        return out

    def prune(self, pruning_method=prune.l1_unstructured, amount=0.2):
        for layer in self.parameters():
            if isinstance(layer, nn.Linear) or isinstance(layer, Conv2d):
                pruning_method(layer, name='weight', amount=amount)

    def perturb_weights(self, perturbation_factor=0.01):
        for layer in self.parameters():
            if isinstance(layer, nn.Linear) or isinstance(layer, Conv2d):
                random_values = torch.randn(layer.size()) * perturbation_factor
                if torch.cuda.is_available():
                    random_values = random_values.cuda()
                layer.weight.data.add_(random_values)
