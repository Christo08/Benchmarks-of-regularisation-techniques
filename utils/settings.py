import torch.nn as nn


class DiabetesSettings:
    batch_size = 58
    categorical_features = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20]
    dropout_layer = [0.18341917, 0.00635388, 0.58905411, 0.394637, 0.62968202]
    learning_rate = 0.02362717591965803
    log_interval = 19
    momentum = 0.03260874342736158
    number_of_epochs = 190
    number_of_fold = 15
    number_of_hidden_layers = 5
    number_of_neurons_in_layers = [150, 60, 200, 30, 20]
    path_to_data = './Data/Numeric/DiabetesHealthIndicators/cleanedData.csv'
    prune_amount = 0.26553199631005675
    prune_epoch_interval = 19
    weight_decay = 0.0001
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


class LiverCirrhosisSettings:
    # not tuning
    path_to_data = 'Data/Numeric/Liver Cirrhosis/cleanedData.csv'
    number_of_fold = 15
    log_interval = 19
    categorical_features = [1, 2, 4, 5, 6, 7]

    # 1st tuning
    batch_size = 1024
    learning_rate = 0.5926620569264897
    momentum = 0.9
    number_of_epochs = 500
    number_of_hidden_layers = 3
    number_of_neurons_in_layers = [800, 850, 100, 100, 200, 100, 800, 300, 1000, 100]

    #2de tuning
    dropout_layer = [0.17116384, 0.24310265, 0.50581393]
    prune_amount = 0.31189753673586107
    prune_epoch_interval = 50
    weight_decay = 0.001
    weight_perturbation_epoch_interval = 105
    weight_perturbation_amount = 0.04433394526141158

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
    categorical_features = []
    dropout_layer = [0.3464047, 0.1661123]
    learning_rate = 0.051741706035525616
    log_interval = 10
    momentum = 0.017682967061624002
    number_of_epochs = 100
    number_of_fold = 15
    number_of_hidden_layers = 2
    number_of_neurons_in_layers = [250, 90, 160, 140, 180]
    path_to_data = 'Data/Numeric/Magic/cleanedData.csv'
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
    categorical_features = []
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
    path_to_data = "Data/Numeric/Mfeat_pixel/cleanedData.csv"

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
    categorical_features = []
    learning_rate = 0.06946966889993425
    log_interval = 35
    number_of_neurons_in_layers = [550, 850, 700, 800, 800, 600, 550, 800, 750, 900]
    number_of_hidden_layers = 7
    momentum = 0.5
    number_of_epochs = 350
    number_of_fold = 3
    path_to_data = "Data/Numeric/Rain in Australia/weatherAUS.csv"
    prune_amount = 0.5
    prune_epoch_interval = 41
    weight_decay = 0.2168418766184487
    weight_perturbation_amount = 0.0001
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
    categorical_features = []
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
    weight_decay = 0.0001
    weight_perturbation_amount = 0.00201
    weight_perturbation_epoch_interval = 33
    path_to_data = 'Data/Numeric/White Wine Quality/cleanedData.csv'

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
    path_to_data = 'Data/Images/Balls/cleanedData'
    in_channels = 3
    output_size = 30
    number_of_fold = 3
    image_size = (224, 224)
    mean = (0.1804, 0.0762, -0.0507)
    std = (0.6568, 0.6530, 0.6908)
    rotation = 360

    # first tuning
    batch_size = 16
    learning_rate = 0.25
    momentum = 0.008515960444937878
    number_of_epochs = 500
    number_of_convolutional_layers = 10
    out_channels = [ 2, 16,  4, 32,  4, 32,  8, 32,  8, 16,  4,  2,  2,  2,  2]
    kernel_size = [2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 4, 2, 2]
    kernel_stride = [16,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  4,  4,  2]
    pool_size =[ 4,  2,  8,  2, 16,  2,  2,  2,  4,  2,  8,  2,  2,  8,  8]
    pool_type =[0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1]
    number_of_hidden_layers = 7
    number_of_neurons_in_layers = [ 300,   50, 1250,  150,  950,  450,  650,  200,   50, 1100]

    # second
    dropout_layer = [0.5, 0.25, 0.1, 0.1, 0.25, 0.4, 0.3, 0.05]
    weight_decay = 0.25
    prune_amount = 0.001
    prune_epoch_interval = 71
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
    path_to_data = 'Data/Images/Bean Leaf Lesions Classification/cleandData'
    in_channels = 3
    output_size = 3
    number_of_fold = 3
    image_size = (500, 500)
    mean = (-0.0299, 0.0369, -0.3763)
    std = (0.4221, 0.4456, 0.4020)
    rotation = 360

    # first tuning
    batch_size = 128
    learning_rate = 0.25
    momentum = 0.0005
    number_of_epochs = 200
    number_of_convolutional_layers = 2
    out_channels = [ 4, 16,  8,  8,  4,  4,  2]
    kernel_size = [2, 2, 2, 4, 2, 2, 2]
    kernel_stride = [32,  2,  2, 16,  2, 32,  2]
    pool_size = [ 4,  2,  2, 64, 32, 16, 64]
    pool_type = [1, 1, 1, 1, 0, 1, 1]
    number_of_hidden_layers = 3
    number_of_neurons_in_layers = [475, 450, 425, 100, 450, 300, 200]

    # second
    dropout_layer = [0.5, 0.25, 0.1, 0.1, 0.25, 0.4, 0.3, 0.05]
    weight_decay = 0.0025621157653690566
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
    path_to_data = 'Data/Images/Cifar-10'
    in_channels = 3
    output_size = 10
    number_of_fold = 3
    log_interval = 10
    mean = (125.3069, 122.9501, 113.8660)
    std = (62.9932, 62.0887, 66.7049)
    image_size = (32, 32)
    rotation = 360

    # frist tuning
    batch_size= 32
    learning_rate= 0.0022766384261316466
    momentum= 0.0979617594743901
    number_of_epochs= 450
    number_of_convolutional_layers= 5
    out_channels= [32,  8, 16, 16, 32, 16, 16, 16,  8,  8,  4, 16,  2,  8,  2]
    kernel_size= [ 4,  4,  2,  2,  2,  2,  2,  4, 32,  2,  4,  8,  2,  4, 16]
    kernel_stride= [ 2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 32, 16,  2,  4,  2]
    pool_size= [ 8,  8,  8,  2,  2,  2,  2,  4,  2, 16,  2,  4, 32,  8,  2]
    pool_type= [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1]
    number_of_hidden_layers= 5
    number_of_neurons_in_layers= [ 550, 1200, 1250, 1350, 1500,  800,   50, 1350, 1500,  300]

    # second
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
    path_to_data = 'Data/Images/MNIST'
    in_channels = 1
    output_size = 10
    number_of_fold = 3
    log_interval = 10
    mean = (0.1307,)
    std = (0.3081,)
    image_size = (28, 28)
    rotation = 360

    # first tuning
    batch_size= 128
    learning_rate= 0.016127805912651637
    momentum= 0.0750431997532428
    number_of_epochs= 350
    number_of_convolutional_layers= 5
    out_channels= [ 3,  2, 17, 15, 19,  2, 21, 23, 16, 15, 20, 10,  6,  5, 10]
    kernel_size= [28,  2,  2,  2,  2,  2,  4, 11,  9, 28, 19, 15, 16, 26, 10]
    kernel_stride= [18,  2,  2,  2,  2,  2,  9,  9,  2,  7,  2, 19,  5, 13, 15]
    pool_size= [ 3,  2,  2,  5,  2, 19, 14, 19,  4, 28, 19, 13,  4, 14,  2]
    pool_type= [1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1]
    number_of_hidden_layers= 10
    number_of_neurons_in_layers= [1000,  700, 1250, 1200, 1150,  700, 1150,  850, 1000,  400]

    # second
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
    path_to_data = 'Data/Images/Shoes/cleandData'
    in_channels = 3
    output_size = 3
    number_of_fold = 3
    log_interval = 10
    image_size = (240, 240)
    mean = (0.4417, 0.4188, 0.3999)
    std = (0.6005, 0.6048, 0.6217)
    rotation = 360

    # first tuning
    batch_size= 16
    learning_rate= 0.02582079536261265
    momentum= 0.026584780497007854
    number_of_epochs= 300
    number_of_convolutional_layers= 2
    out_channels= [64,  8,  4,  2,  2,  2, 64]
    kernel_size= [16,  2,  2,  8,  4, 64,  4]
    kernel_stride= [4, 2, 2, 2, 4, 2, 2]
    pool_size= [ 4,  2,  2,  2, 16,  4,  4]
    pool_type= [0, 1, 0, 1, 0, 0, 0]
    number_of_hidden_layers= 2,
    number_of_neurons_in_layers= [375, 300, 350, 475, 150, 325, 500]

    # second
    dropout_layer = [0.5, 0.25, 0.1, 0.1, 0.25, 0.4, 0.3, 0.05]
    weight_decay = 0.001
    prune_amount = 0.028165123278217293
    prune_epoch_interval = 100
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
