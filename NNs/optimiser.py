import numpy as np
import pyhopper

from NNs.Images.baseline import run as baseline_image_run
from NNs.Images.dropout import run as dropout_image_run
from NNs.Images.pruning import run as pruning_image_run
from NNs.Images.weightNormalisation import run as weight_normalisation_image_run
from NNs.Images.weightPerturbation import run as weight_perturbation_image_run
from NNs.Numeric.baseline import run as baseline_numeric_run
from NNs.Numeric.dropout import run as dropout_numeric_run
from NNs.Numeric.pruning import run as pruning_numeric_run
from NNs.Numeric.weightNormalisation import run as weight_normalisation_numeric_run
from NNs.Numeric.weightPerturbation import run as weight_perturbation_numeric_run
from utils.dataLoader import load_numeric_data_set, load_images_datas_set
from utils.menus import *

MAX_NUMBER_OF_LAYERS = 10
MIN_NUMBER_OF_LAYERS = 2
MAX_NUMBER_OF_EPOCH = 1000
STEPS = 10

basic_NN_parameters = {
    "batch_size": pyhopper.int(16, 1024, power_of=2),
    "learning_rate":  pyhopper.float(0.0001,0.5,"0.4f"),
    "momentum": pyhopper.float(0.0001,0.5,"0.4f"),
    "number_of_epochs": pyhopper.int(50, MAX_NUMBER_OF_EPOCH, multiple_of=20),
    "number_of_hidden_layers": pyhopper.int(MIN_NUMBER_OF_LAYERS, MAX_NUMBER_OF_LAYERS),
    "number_of_neurons_in_layers": pyhopper.int(5, 1000, multiple_of=5, shape=MAX_NUMBER_OF_LAYERS)
}

basic_CNN_parameters = {
    "batch_size": pyhopper.int(16, 128, power_of=2),
    "learning_rate": pyhopper.float(0.0005, 0.25, log=True),
    "momentum": pyhopper.float(0.0005, 0.25, log=True),
    "number_of_epochs": pyhopper.int(50, MAX_NUMBER_OF_EPOCH, multiple_of=50),
    "number_of_convolutional_layers": pyhopper.int(1, 10),
    "out_channels": pyhopper.int(32, 64, power_of=2, shape=10),

    "kernel_size": pyhopper.int(1, 10, shape=10),
    "kernel_stride": pyhopper.int(1, 10, shape=10),
    "padding": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    "pool_size": pyhopper.int(1, 10, shape=10),
    "pool_type": pyhopper.int(0, 1, shape=10),

    "number_of_hidden_layers": pyhopper.int(MIN_NUMBER_OF_LAYERS, MAX_NUMBER_OF_LAYERS),
    "number_of_neurons_in_layers": pyhopper.int(5, 1000, multiple_of=5, shape=MAX_NUMBER_OF_LAYERS),
}

dropout_parameters = {
    "dropout_layers": pyhopper.float(0.01, 0.75, shape=MAX_NUMBER_OF_LAYERS)
}

prune_parameters = {
    "prune_amount": pyhopper.float(0.01, 0.75, "0.2f"),
    "prune_epoch_interval": pyhopper.int(4, 20)
}

weight_decay_parameters = {
    "weight_decay": pyhopper.float(0.01, 0.5, "0.2f")
}

weight_perturbation_parameters = {
    "weight_perturbation_amount": pyhopper.float(0.01, 1.00, "0.2f"),
    "weight_perturbation_interval": pyhopper.int(MAX_NUMBER_OF_EPOCH / 250, MAX_NUMBER_OF_EPOCH / 10)
}

dataset_name = ""
basic_settings = {}
training_set = ""
validation_set = ""

def optimiser():
    while True:
        module_types = show_model_menu()
        if len(module_types) == 0:
            break
        while True:
            regularisation_method = show_optimiser_regularisation_menu()
            if len(regularisation_method) == 0:
                break
            datasets = show_dataset_menu("ALL" if len(module_types) == len(MODULE_TYPES) else module_types[0])
            if len(datasets) == 0:
                continue
            for dataset in datasets:
                if dataset in IMAGE_DATASETS:
                    optimise_cnn(regularisation_method, dataset)
                else:
                    optimise_nn(regularisation_method, dataset)

def optimise_nn(regularisation_method, dataset_name):
    global technique, dataset, setting, training_set, validation_set
    dataset = dataset_name
    training_set, validation_set, setting = load_numeric_data_set(dataset_name)
    for technique_name in regularisation_method:
        technique = technique_name
        if technique == OPTIMISER_REGULAR_TYPES[0]:
            search = pyhopper.Search(basic_NN_parameters)
        elif technique == OPTIMISER_REGULAR_TYPES[1]:
            search = pyhopper.Search(dropout_parameters)
        elif technique == OPTIMISER_REGULAR_TYPES[2]:
            search = pyhopper.Search(prune_parameters)
        elif technique == OPTIMISER_REGULAR_TYPES[3]:
            search = pyhopper.Search(weight_decay_parameters)
        elif technique == OPTIMISER_REGULAR_TYPES[4]:
            search = pyhopper.Search(weight_perturbation_parameters)
        best_params = search.run(
            nn_trainer_wrapper,
            direction="min",
            steps=STEPS,
            # n_jobs="per-gpu"
        )
        validation_losses = nn_trainer_wrapper(best_params)
        print(f"Tuned params test {dataset} loss: {validation_losses:0.2f}%")
        print(dataset + ": ", best_params)

def optimise_cnn(regularisation_method, dataset_name):
    global technique, dataset, setting, training_set, validation_set
    dataset = dataset_name
    training_set, validation_set, setting = load_images_datas_set(dataset_name, False)
    for technique_name in regularisation_method:
        technique = technique_name
        if technique == OPTIMISER_REGULAR_TYPES[0]:
            search = pyhopper.Search(basic_CNN_parameters)
        elif technique == OPTIMISER_REGULAR_TYPES[1]:
            search = pyhopper.Search(dropout_parameters)
        elif technique == OPTIMISER_REGULAR_TYPES[2]:
            search = pyhopper.Search(prune_parameters)
        elif technique == OPTIMISER_REGULAR_TYPES[3]:
            search = pyhopper.Search(weight_decay_parameters)
        elif technique == OPTIMISER_REGULAR_TYPES[4]:
            search = pyhopper.Search(weight_perturbation_parameters)
        best_params = search.run(
            cnn_trainer_wrapper,
            direction="min",
            steps=STEPS,
            # n_jobs="per-gpu"
        )
        validation_losses = cnn_trainer_wrapper(best_params)
        print(f"Tuned params test {dataset} loss: {validation_losses:0.2f}%")
        print(dataset + ": ", best_params)

def nn_trainer_wrapper(params):
    global technique, dataset, setting, training_set, validation_set
    setting = setting.json_to_object(params)
    if technique == OPTIMISER_REGULAR_TYPES[0]:
        result = baseline_numeric_run(dataset, setting, training_set, validation_set, False)
    elif technique == OPTIMISER_REGULAR_TYPES[1]:
        result = dropout_numeric_run(dataset, setting, training_set, validation_set, False)
    elif technique == OPTIMISER_REGULAR_TYPES[2]:
        result = pruning_numeric_run(dataset, setting, training_set, validation_set, False)
    elif technique == OPTIMISER_REGULAR_TYPES[3]:
        result = weight_normalisation_numeric_run(dataset, setting, training_set, validation_set, False)
    else:
        result = weight_perturbation_numeric_run(dataset, setting, training_set, validation_set, False)

    return  np.mean(result["results"]["losses"]["testing"][-1])

def cnn_trainer_wrapper(params):
    global technique, dataset, setting, training_set, validation_set
    setting = setting.json_to_object(params)
    if technique == OPTIMISER_REGULAR_TYPES[0]:
        result = baseline_image_run(dataset, setting, training_set, validation_set, False)
    elif technique == OPTIMISER_REGULAR_TYPES[1]:
        result = dropout_image_run(dataset, setting, training_set, validation_set, False)
    elif technique == OPTIMISER_REGULAR_TYPES[2]:
        result = pruning_image_run(dataset, setting, training_set, validation_set, False)
    elif technique == OPTIMISER_REGULAR_TYPES[3]:
        result = weight_normalisation_image_run(dataset, setting, training_set, validation_set, False)
    else:
        result = weight_perturbation_image_run(dataset, setting, training_set, validation_set, False)

    return  np.mean(result["results"]["losses"]["testing"][-1])