import torch.nn as nn


class DiabetesSettings:
    batch_size = 58
    dropout_layer = [0.18341917, 0.00635388, 0.58905411, 0.394637, 0.62968202]
    learning_rate = 0.02362717591965803
    log_interval = 19
    momentum = 0.03260874342736158
    number_of_epochs = 190
    number_of_fold = 15
    number_of_hidden_layers = 5
    number_of_neurons_in_layers = [150, 60, 200, 30, 20]
    path_to_data = 'C:\\Users\\User\\OneDrive\\tuks\\master\\code\\Data\\Numeric\\DiabetesHealthIndicators\\cleanedData.csv'
    prune_amount = 0.26553199631005675
    prune_epoch_interval = 19
    weight_decay = 0.2753287741235434
    weight_perturbation_epoch_interval = 22
    weight_perturbation_amount = 0.001

    def to_json_serializable(self):
        return {
            "batch_size": self.batch_size,
            "dropout_layer": self.dropout_layer,
            "learning_rate": self.learning_rate,
            "log_interval": self.log_interval,
            "momentum": self.momentum,
            "number_of_epochs": self.number_of_epochs,
            "number_of_fold": self.number_of_fold,
            "number_of_hidden_layers": self.number_of_hidden_layers,
            "number_of_neurons_in_layers": self.number_of_neurons_in_layers,
            "path_to_data": self.path_to_data,
            "prune_amount": self.prune_amount,
            "prune_epoch_interval": self.prune_epoch_interval,
            "weight_decay": self.weight_decay,
            "weight_perturbation_amount": self.weight_perturbation_amount,
            "weight_perturbation_epoch_interval": self.weight_perturbation_epoch_interval
        }


class MagicSettings:
    batch_size = 56
    dropout_layer = [0.3464047, 0.1661123]
    learning_rate = 0.051741706035525616
    log_interval = 10
    momentum = 0.017682967061624002
    number_of_epochs = 100
    number_of_fold = 15
    number_of_hidden_layers = 2
    number_of_neurons_in_layers = [250, 90, 160, 140, 180]
    path_to_data = 'C:\\Users\\User\\OneDrive\\tuks\\master\\code\\Data\\Numeric\\Magic\\cleanedData.csv'
    prune_amount = 0.3024614034723992
    prune_epoch_interval = 10
    weight_decay = 0.0021290144020112553
    weight_perturbation_epoch_interval = 17
    weight_perturbation_amount = 0.001

    def to_json_serializable(self):
        return {
            "batch_size": self.batch_size,
            "dropout_layer": self.dropout_layer,
            "learning_rate": self.learning_rate,
            "log_interval": self.log_interval,
            "momentum": self.momentum,
            "number_of_epochs": self.number_of_epochs,
            "number_of_fold": self.number_of_fold,
            "number_of_hidden_layers": self.number_of_hidden_layers,
            "number_of_neurons_in_layers": self.number_of_neurons_in_layers,
            "path_to_data": self.path_to_data,
            "prune_amount": self.prune_amount,
            "prune_epoch_interval": self.prune_epoch_interval,
            "weight_decay": self.weight_decay,
            "weight_perturbation_amount": self.weight_perturbation_amount,
            "weight_perturbation_epoch_interval": self.weight_perturbation_epoch_interval
        }


class MfeatPixelSettings:
    batch_size = 42
    dropout_layer = [0.37802851, 0.51423422, 0.28836089, 0.73270707]
    learning_rate = 0.030189840897776998
    log_interval = 50
    number_of_neurons_in_layers = [180, 120, 70, 160, 180]
    number_of_fold = 15
    number_of_hidden_layers = 4
    momentum = 0.07134776585219803
    number_of_epochs = 500
    prune_amount = 0.19899206387087048
    prune_epoch_interval = 100
    weight_decay = 0.0017247976613359497
    weight_perturbation_amount = 0.001
    weight_perturbation_epoch_interval = 3
    path_to_data = "C:\\Users\\User\\OneDrive\\tuks\\master\\code\\Data\\Numeric\\Mfeat_pixel\\cleanedData.csv"

    def to_json_serializable(self):
        return {
            "batch_size": self.batch_size,
            "dropout_layer": self.dropout_layer,
            "learning_rate": self.learning_rate,
            "log_interval": self.log_interval,
            "momentum": self.momentum,
            "number_of_epochs": self.number_of_epochs,
            "number_of_fold": self.number_of_fold,
            "number_of_hidden_layers": self.number_of_hidden_layers,
            "number_of_neurons_in_layers": self.number_of_neurons_in_layers,
            "path_to_data": self.path_to_data,
            "prune_amount": self.prune_amount,
            "prune_epoch_interval": self.prune_epoch_interval,
            "weight_decay": self.weight_decay,
            "weight_perturbation_amount": self.weight_perturbation_amount,
            "weight_perturbation_epoch_interval": self.weight_perturbation_epoch_interval
        }


class RainInAustraliaSettings:
    batch_size = 88
    dropout_layer = [0.47023132, 0.2833416, 0.44631075, 0.400374, 0.47384847, 0.59316255, 0.47270801]
    learning_rate = 0.06946966889993425
    log_interval = 35
    number_of_neurons_in_layers = [550, 850, 700, 800, 800, 600, 550, 800, 750, 900, 750, 750, 650, 850, 850, 750, 750,
                                   700, 600,
                                   650]
    number_of_hidden_layers = 7
    momentum = 0.5
    number_of_epochs = 350
    number_of_fold = 15
    path_to_data = "C:\\Users\\User\\OneDrive\\tuks\\master\\code\\Data\\Numeric\\Rain in Australia\\cleanedData.csv"
    prune_amount = 0.5
    prune_epoch_interval = 41
    weight_decay = 0.2168418766184487
    weight_perturbation_amount = 0.001
    weight_perturbation_epoch_interval = 33

    def to_json_serializable(self):
        return {
            "batch_size": self.batch_size,
            "dropout_layer": self.dropout_layer,
            "learning_rate": self.learning_rate,
            "log_interval": self.log_interval,
            "momentum": self.momentum,
            "number_of_epochs": self.number_of_epochs,
            "number_of_fold": self.number_of_fold,
            "number_of_hidden_layers": self.number_of_hidden_layers,
            "number_of_neurons_in_layers": self.number_of_neurons_in_layers,
            "path_to_data": self.path_to_data,
            "prune_amount": self.prune_amount,
            "prune_epoch_interval": self.prune_epoch_interval,
            "weight_decay": self.weight_decay,
            "weight_perturbation_amount": self.weight_perturbation_amount,
            "weight_perturbation_epoch_interval": self.weight_perturbation_epoch_interval
        }


class WhiteWineQualitySettings:
    batch_size = 32
    dropout_layer = [0, 0, 0.01221672, 0.57899157, 0.56500733]
    learning_rate = 0.00811431704457397
    log_interval = 20
    number_of_neurons_in_layers = [300, 250, 200, 850, 400, 4, 200, 1000, 700, 400, 350, 1000, 550, 900, 450, 150, 4,
                                   400, 4, 650]
    number_of_hidden_layers = 5
    momentum = 0.5769796824824637
    number_of_epochs = 220
    number_of_fold = 15
    prune_amount = 0.07553485715290663
    prune_epoch_interval = 38
    weight_decay = 0.3938271005258168
    weight_perturbation_amount = 0.001
    weight_perturbation_epoch_interval = 33
    path_to_data = 'C:\\Users\\User\\OneDrive\\tuks\\master\\code\\Data\\Numeric\\White Wine Quality\\cleanedData.csv'

    def to_json_serializable(self):
        return {
            "batch_size": self.batch_size,
            "dropout_layer": self.dropout_layer,
            "learning_rate": self.learning_rate,
            "log_interval": self.log_interval,
            "momentum": self.momentum,
            "number_of_epochs": self.number_of_epochs,
            "number_of_fold": self.number_of_fold,
            "number_of_hidden_layers": self.number_of_hidden_layers,
            "number_of_neurons_in_layers": self.number_of_neurons_in_layers,
            "path_to_data": self.path_to_data,
            "prune_amount": self.prune_amount,
            "prune_epoch_interval": self.prune_epoch_interval,
            "weight_decay": self.weight_decay,
            "weight_perturbation_amount": self.weight_perturbation_amount,
            "weight_perturbation_epoch_interval": self.weight_perturbation_epoch_interval
        }


class BallsSettings:
    # not tuning
    log_interval = 10
    path_to_data = 'C:\\Users\\User\\OneDrive\\tuks\\master\\code\\Data\\Images\\Balls\\cleanedData'
    in_channels = 3
    output_size = 30
    number_of_fold = 3
    image_size = (224, 224)
    mean = (0.1804, 0.0762, -0.0507)
    std = (0.6568, 0.6530, 0.6908)

    # first tuning
    batch_size = 64
    learning_rate = 0.03274305144410664
    momentum = 0.0005000000000000001
    number_of_epochs = 500
    number_of_convolutional_layers = 1
    out_channels = [64, 8, 2, 32, 4, 64, 32, 8, 8, 4]
    kernel_size = [8, 2, 4, 4, 2, 4, 32, 4, 2, 16]
    kernel_stride = [2, 2, 2, 2, 16, 2, 4, 2, 8, 16]
    pool_size = [4, 2, 16, 4, 4, 2, 4, 8, 2, 2]
    pool_type = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    number_of_hidden_layers = 1
    number_of_neurons_in_layers = [150, 200, 300, 300, 850]

    # second
    rotation = 0
    dropout_layer = [0.5, 0.25, 0.1, 0.1, 0.25, 0.4, 0.3, 0.05]
    weight_decay = 0.25
    prune_amount = 0.07553485715290663
    prune_epoch_interval = 38
    weight_perturbation_amount = 0.001
    weight_perturbation_epoch_interval = 33

    def to_json_serializable(self):
        return {
            "log_interval": self.log_interval,
            "path_to_data": self.path_to_data,
            "in_channels": self.in_channels,
            "output_size": self.output_size,
            "number_of_fold": self.number_of_fold,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "pool_size": self.pool_size,
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "number_of_epochs": self.number_of_epochs,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "batch_size": self.batch_size,
            "rotation": self.rotation,
            "dropout_layer": self.dropout_layer,
            "weight_decay": self.weight_decay,
            "prune_amount": self.prune_amount,
            "prune_epoch_interval": self.prune_epoch_interval,
            "weight_perturbation_amount": self.weight_perturbation_amount,
            "weight_perturbation_epoch_interval": self.weight_perturbation_epoch_interval
        }


class BeanLeafSettings:
    log_interval = 10
    path_to_data = 'C:\\Users\\User\\OneDrive\\tuks\\master\\code\\Data\\Images\\Bean Leaf Lesions Classification\\cleandData'
    in_channels = 3
    output_size = 3
    number_of_fold = 3
    image_size = (500, 500)
    mean = (-0.0299, 0.0369, -0.3763)
    std = (0.4221, 0.4456, 0.4020)

    # first tuning
    batch_size = 32
    learning_rate = 0.00303644029210212
    momentum = 0.0625482730489677
    number_of_epochs = 300
    number_of_convolutional_layers = 1
    out_channels = [16, 16, 16, 8, 2, 16, 8, 2, 2, 2]
    kernel_size = [64, 2, 2, 8, 2, 2, 64, 64, 4, 2]
    kernel_stride = [16, 2, 2, 4, 32, 4, 4, 8, 64, 16]
    pool_size = [4, 8, 8, 2, 32, 16, 8, 32, 4, 8]
    pool_type = [1, 1, 0, 1, 0, 0, 1, 0, 1, 0]
    number_of_hidden_layers = 2
    number_of_neurons_in_layers = [600, 1000, 750, 550, 750]

    # second
    rotation = 0
    dropout_layer = [0.5, 0.25, 0.1, 0.1, 0.25, 0.4, 0.3, 0.05]
    weight_decay = 0.25
    prune_amount = 0.07553485715290663
    prune_epoch_interval = 38
    weight_perturbation_amount = 0.001
    weight_perturbation_epoch_interval = 33

    def to_json_serializable(self):
        return {
            "log_interval": self.log_interval,
            "path_to_data": self.path_to_data,
            "in_channels": self.in_channels,
            "output_size": self.output_size,
            "number_of_fold": self.number_of_fold,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "pool_size": self.pool_size,
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "number_of_epochs": self.number_of_epochs,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "batch_size": self.batch_size,
            "rotation": self.rotation,
            "dropout_layer": self.dropout_layer,
            "weight_decay": self.weight_decay,
            "prune_amount": self.prune_amount,
            "prune_epoch_interval": self.prune_epoch_interval,
            "weight_perturbation_amount": self.weight_perturbation_amount,
            "weight_perturbation_epoch_interval": self.weight_perturbation_epoch_interval
        }


class CifarSettings:
    path_to_data = 'C:\\Users\\User\\OneDrive\\tuks\\master\\code\\Data\\Images\\Cifar-10'
    in_channels = 3
    output_size = 10
    number_of_fold = 3
    log_interval = 10
    mean = (125.3069, 122.9501, 113.8660)
    std = (62.9932, 62.0887, 66.7049)
    image_size = (32, 32)

    # frist tuning
    out_channels = [64, 128, 256, 512]
    kernel_size = [7, 5, 5, 3]
    stride = [10, 5, 5, 3]
    pool_size = [(25, 5), (25, 5), (9, 3), (1, 1)]
    hidden_layer_sizes = [512, 320, 80, 40]
    number_of_epochs = 100
    learning_rate = 0.01
    momentum = 0.5
    batch_size = 32

    # second
    rotation = 0
    dropout_layer = [0.5, 0.25, 0.1, 0.1, 0.25, 0.4, 0.3, 0.05]
    weight_decay = 0.25
    prune_amount = 0.07553485715290663
    prune_epoch_interval = 38
    weight_perturbation_amount = 0.001
    weight_perturbation_epoch_interval = 33

    def to_json_serializable(self):
        return {
            "log_interval": self.log_interval,
            "path_to_data": self.path_to_data,
            "in_channels": self.in_channels,
            "output_size": self.output_size,
            "number_of_fold": self.number_of_fold,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "pool_size": self.pool_size,
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "number_of_epochs": self.number_of_epochs,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "batch_size": self.batch_size,
            "rotation": self.rotation,
            "dropout_layer": self.dropout_layer,
            "weight_decay": self.weight_decay,
            "prune_amount": self.prune_amount,
            "prune_epoch_interval": self.prune_epoch_interval,
            "weight_perturbation_amount": self.weight_perturbation_amount,
            "weight_perturbation_epoch_interval": self.weight_perturbation_epoch_interval
        }


class MNISTSettings:
    path_to_data = 'C:\\Users\\User\\OneDrive\\tuks\\master\\code\\Data\\Images\\MNIST'
    in_channels = 1
    output_size = 10
    number_of_fold = 3
    log_interval = 10
    mean = (0.1307,)
    std = (0.3081,)
    image_size = (28, 28)

    # frist tuning
    out_channels = [64, 128, 256, 512]
    kernel_size = [7, 5, 5, 3]
    stride = [10, 5, 5, 3]
    pool_size = [(25, 5), (25, 5), (9, 3), (1, 1)]
    hidden_layer_sizes = [512, 320, 80, 40]
    number_of_epochs = 100
    learning_rate = 0.01
    momentum = 0.5
    batch_size = 32

    # second
    rotation = 0
    dropout_layer = [0.5, 0.25, 0.1, 0.1, 0.25, 0.4, 0.3, 0.05]
    weight_decay = 0.25
    prune_amount = 0.07553485715290663
    prune_epoch_interval = 38
    weight_perturbation_amount = 0.001
    weight_perturbation_epoch_interval = 33

    def to_json_serializable(self):
        return {
            "log_interval": self.log_interval,
            "path_to_data": self.path_to_data,
            "in_channels": self.in_channels,
            "output_size": self.output_size,
            "number_of_fold": self.number_of_fold,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "pool_size": self.pool_size,
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "number_of_epochs": self.number_of_epochs,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "batch_size": self.batch_size,
            "rotation": self.rotation,
            "dropout_layer": self.dropout_layer,
            "weight_decay": self.weight_decay,
            "prune_amount": self.prune_amount,
            "prune_epoch_interval": self.prune_epoch_interval,
            "weight_perturbation_amount": self.weight_perturbation_amount,
            "weight_perturbation_epoch_interval": self.weight_perturbation_epoch_interval
        }


class ShoesSettings:
    path_to_data = 'C:\\Users\\User\\OneDrive\\tuks\\master\\code\\Data\\Images\\Shoes\\cleandData'
    in_channels = 3
    output_size = 3
    number_of_fold = 3
    log_interval = 10
    image_size = (240, 240)
    mean = (0.4417, 0.4188, 0.3999)
    std = (0.6005, 0.6048, 0.6217)

    # first tuning
    batch_size = 32
    learning_rate = 0.0087804470862461
    momentum = 0.0755492952986247
    number_of_epochs = 500
    number_of_convolutional_layers = 1
    out_channels = [64, 2, 8, 2, 2, 4, 2, 4, 4, 32]
    kernel_size = [8, 2, 2, 2, 8, 4, 2, 8, 2, 64]
    kernel_stride = [2, 2, 2, 2, 4, 32, 4, 32, 16, 8]
    pool_size = [4, 2, 8, 4, 4, 32, 4, 4, 16, 2]
    pool_type = [0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
    number_of_hidden_layers = 4
    number_of_neurons_in_layers = [150, 900, 250, 800, 450]

    # second
    rotation = 0
    dropout_layer = [0.5, 0.25, 0.1, 0.1, 0.25, 0.4, 0.3, 0.05]
    weight_decay = 0.25
    prune_amount = 0.07553485715290663
    prune_epoch_interval = 38
    weight_perturbation_amount = 0.001
    weight_perturbation_epoch_interval = 33

    def to_json_serializable(self):
        return {
            "log_interval": self.log_interval,
            "path_to_data": self.path_to_data,
            "in_channels": self.in_channels,
            "output_size": self.output_size,
            "number_of_fold": self.number_of_fold,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "pool_size": self.pool_size,
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "number_of_epochs": self.number_of_epochs,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "batch_size": self.batch_size,
            "rotation": self.rotation,
            "dropout_layer": self.dropout_layer,
            "weight_decay": self.weight_decay,
            "prune_amount": self.prune_amount,
            "prune_epoch_interval": self.prune_epoch_interval,
            "weight_perturbation_amount": self.weight_perturbation_amount,
            "weight_perturbation_epoch_interval": self.weight_perturbation_epoch_interval
        }
