import torch
import torch.nn as nn
from torch.nn import Conv2d
from torch.nn.utils import prune


class Net(nn.Module):
    def __init__(self,
                 input_image_size,
                 in_channels,
                 number_of_convolutional_layers,
                 out_channels,
                 padding,
                 kernel_size,
                 kernel_stride,
                 pool_size,
                 pool_type,
                 number_of_hidden_layers,
                 number_of_neurons_in_layers,
                 output_size,
                 batch_norm=False,
                 dropout=None,
                 layer_norm=False,
                 weight_norm=False):
        super(Net, self).__init__()
        cnn_layers = []
        dropout_index = 0
        for counter in range(number_of_convolutional_layers):
            if counter == 0:
                conv_layer = Conv2d(in_channels=in_channels, out_channels=out_channels[counter],
                                    kernel_size=kernel_size[counter], stride=kernel_stride[counter], padding=padding[counter],
                                    bias=False)
            else:
                conv_layer = Conv2d(in_channels=out_channels[counter - 1], out_channels=out_channels[counter],
                                    kernel_size=kernel_size[counter], stride=kernel_stride[counter], padding=padding[counter],
                                    bias=False)
            if weight_norm:
                conv_layer = torch.nn.utils.parametrizations.weight_norm(conv_layer)

            cnn_layers.append(conv_layer)

            if pool_type[counter] == 0:
                cnn_layers.append(nn.MaxPool2d(kernel_size=pool_size[counter], stride=pool_size[counter], padding=0))
            else:
                cnn_layers.append(nn.AvgPool2d(kernel_size=pool_size[counter], stride=pool_size[counter], padding=0))

            cnn_layers.append(nn.LeakyReLU(inplace=True))

            if batch_norm:
                cnn_layers.append(nn.BatchNorm2d(out_channels[counter]))

            if layer_norm:
                conv_output_size = input_image_size
                conv_output_size = self.calculate_output_dim(conv_output_size, cnn_layers)
                cnn_layers.append(nn.LayerNorm([out_channels[counter], conv_output_size[0], conv_output_size[1]]))

            if dropout is not None and dropout[dropout_index] > 0:
                cnn_layers.append(nn.Dropout2d(p=dropout[dropout_index]))
            dropout_index += 1

        self.cnn_layers = nn.Sequential(*cnn_layers)

        conv_output_size = input_image_size
        conv_output_size = self.calculate_output_dim(conv_output_size, cnn_layers)
        input_size = out_channels[number_of_convolutional_layers-1] * conv_output_size[0] * conv_output_size[1]

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

        if dropout is not None and dropout[dropout_index] > 0:
            layers.append(nn.Dropout1d(p=dropout[dropout_index]))
        dropout_index += 1

        if number_of_hidden_layers > 1:
            for counter in range(number_of_hidden_layers - 1):
                layer = nn.Linear(number_of_neurons_in_layers[counter], number_of_neurons_in_layers[counter + 1])
                if weight_norm:
                    layer = torch.nn.utils.parametrizations.weight_norm(layer)
                layers.append(layer)

                layers.append(nn.LeakyReLU())

                if batch_norm:
                    layers.append(nn.BatchNorm1d(number_of_neurons_in_layers[counter + 1]))  # Adjusted batch_norm size

                if layer_norm:
                    layers.append(nn.LayerNorm(number_of_neurons_in_layers[counter + 1]))

                if dropout is not None and dropout[dropout_index] > 0:
                    layers.append(nn.Dropout1d(p=dropout[dropout_index]))
                dropout_index += 1
        # Add the output layer
        layers.append(nn.Linear(number_of_neurons_in_layers[number_of_hidden_layers - 1], output_size))
        layers.append(nn.Softmax(dim=1))
        self.layers = nn.Sequential(*layers)

    def calculate_output_dim(self, input_size, layers):
            for layer in layers:
                if isinstance(layer, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
                    kernel = layer.kernel_size if isinstance(layer.kernel_size, tuple) else (layer.kernel_size, layer.kernel_size)
                    stride = layer.stride if isinstance(layer.stride, tuple) else (layer.stride, layer.stride)
                    padding = layer.padding if isinstance(layer.padding, tuple) else (layer.padding, layer.padding)
                    input_size = (((input_size[0] - kernel[0] + 2 * padding[0]) // stride[0]) + 1,
                                  ((input_size[1] - kernel[1] + 2 * padding[1]) // stride[1]) + 1)


            return input_size

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x

    def prune(self, pruning_method=prune.l1_unstructured, amount=0.2):
        for layer in self.parameters():
            if isinstance(layer, nn.Linear) or isinstance(layer, Conv2d):
                pruning_method(layer, name='weight', amount=amount)

    def perturb_weights(self, perturbation_factor=0.01):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                random_values = torch.randn(module.weight.size()) * perturbation_factor
                if module.weight.is_cuda:
                    random_values = random_values.cuda()
                module.weight.data.add_(random_values)