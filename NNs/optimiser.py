import numpy as np
import pyhopper
from pyhopper import search

from utils.dataLoader import load_numeric_data_set, load_images_datas_set
from utils.menus import show_model_menu, MODULE_TYPES, NUMERIC_REGULAR_TYPES, show_regularisation_menu, \
    show_dataset_menu, IMAGE_DATASETS, NUMERIC_DATASETS, IMAGE_REGULAR_TYPES

from NNs.Images.baseline import run as baseline_images_run
from NNs.Images.batchNormalisation import run as batch_normalisation_images_run
from NNs.Images.dropout import run as dropout_images_run
from NNs.Images.geometricTransformation import run as geometric_transformation_images_run
from NNs.Images.layerNormalisation import run as layer_normalisation_images_run
from NNs.Images.pruning import run as pruning_images_run
from NNs.Images.regularisationTerm import run as regularisation_term_images_run
from NNs.Images.weightNormalisation import run as weight_normalisation_images_run
from NNs.Images.weightPerturbation import run as weight_perturbation_images_run

from NNs.Numeric.baseline import run as baseline_numeric_run
from NNs.Numeric.batchNormalisation import run as batch_normalisation_numeric_run
from NNs.Numeric.dropout import run as dropout_numeric_run
from NNs.Numeric.layerNormalisation import run as layer_normalisation_numeric_run
from NNs.Numeric.pruning import run as pruning_numeric_run
from NNs.Numeric.regularisationTerm import run as regularisation_term_numeric_run
from NNs.Numeric.smote import run as SMOTE_numeric_run
from NNs.Numeric.weightNormalisation import run as weight_normalisation_numeric_run
from NNs.Numeric.weightPerturbation import run as weight_perturbation_numeric_run
from utils.settings import DiabetesSettings, LiverCirrhosisSettings, MagicSettings, MfeatPixelSettings, \
    WhiteWineQualitySettings, BallsSettings, BeanLeafSettings, FashionMNISTSettings, CifarSettings, MNISTSettings, \
    ShoesSettings

MAX_NUMBER_OF_LAYERS = 10
MIN_NUMBER_OF_LAYERS = 2
MAX_NUMBER_OF_EPOCH = 1000

basic_parameters = {
    "batch_size": pyhopper.int(16, 1024, power_of=2),
    "learning_rate":  pyhopper.float(0.0001,0.5,"0.4f"),
    "momentum": pyhopper.float(0.0001,0.5,"0.4f"),
    "number_of_epochs": pyhopper.int(50, MAX_NUMBER_OF_EPOCH, multiple_of=20),
    "number_of_hidden_layers": pyhopper.int(MIN_NUMBER_OF_LAYERS, MAX_NUMBER_OF_LAYERS),
    "number_of_neurons_in_layers": pyhopper.int(5, 1000, multiple_of=5, shape=MAX_NUMBER_OF_LAYERS)
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
            regularisation_method = show_regularisation_menu("BOTH" if len(module_types) == len(MODULE_TYPES) else module_types[0])
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
    setting = setting.to_json_serializable()
    for technique_name in regularisation_method:
        technique = technique_name
        if technique == NUMERIC_REGULAR_TYPES[0]:
            search = pyhopper.Search(basic_parameters)
        else:
            if technique == NUMERIC_REGULAR_TYPES[1]:
                search = pyhopper.Search(dropout_parameters)
            elif technique == NUMERIC_REGULAR_TYPES[2]:
                search = pyhopper.Search(prune_parameters)
            elif technique == NUMERIC_REGULAR_TYPES[3]:
                search = pyhopper.Search(weight_decay_parameters)
            elif technique == NUMERIC_REGULAR_TYPES[4]:
                search = pyhopper.Search(weight_perturbation_parameters)
        best_params = search.run(
            nn_trainer_wrapper,
            direction="min",
            steps=10,
            # n_jobs="per-gpu"
        )
        validation_losses = nn_trainer_wrapper(best_params)
        print(f"Tuned params test {dataset} loss: {validation_losses:0.2f}%")
        print(dataset + ": ", best_params)

def optimise_cnn(regularisation_method, dataset_name):
    pass

def nn_trainer_wrapper(params):
    global technique, dataset, setting, training_set, validation_set
    if technique == NUMERIC_REGULAR_TYPES[0]:
        full_settings = jsonSettongsToObjects(dataset, params)
        result = baseline_numeric_run(dataset, full_settings, training_set, validation_set, False)
    else:
        full_settings = {**setting, **params}
        full_settings = jsonSettongsToObjects(dataset, full_settings)
        if technique == NUMERIC_REGULAR_TYPES[1]:
            result = dropout_numeric_run(dataset, full_settings, training_set, validation_set, False)
        elif technique == NUMERIC_REGULAR_TYPES[2]:
            result = pruning_numeric_run(dataset, full_settings, training_set, validation_set, False)
        elif technique == NUMERIC_REGULAR_TYPES[3]:
            result = weight_normalisation_numeric_run(dataset, full_settings, training_set, validation_set, False)
        else:
            result = weight_perturbation_numeric_run(dataset, full_settings, training_set, validation_set, False)

    return  np.mean(result["results"]["losses"]["testing"][-1])

def jsonSettongsToObjects(dataset_name, params):
    if dataset_name == NUMERIC_DATASETS[0]:
        return DiabetesSettings().json_to_object(params)
    elif dataset_name == NUMERIC_DATASETS[1]:
        return LiverCirrhosisSettings().json_to_object(params)
    elif dataset_name == NUMERIC_DATASETS[2]:
        return MagicSettings().json_to_object(params)
    elif dataset_name == NUMERIC_DATASETS[3]:
        return MfeatPixelSettings().json_to_object(params)
    elif dataset_name == NUMERIC_DATASETS[4]:
        return WhiteWineQualitySettings().json_to_object(params)

    elif dataset_name == IMAGE_REGULAR_TYPES[0]:
        return BallsSettings().json_to_object(params)
    elif dataset_name == IMAGE_REGULAR_TYPES[1]:
        return BeanLeafSettings().json_to_object(params)
    elif dataset_name == IMAGE_REGULAR_TYPES[2]:
        return FashionMNISTSettings().json_to_object(params)
    elif dataset_name == IMAGE_REGULAR_TYPES[3]:
        return CifarSettings().json_to_object(params)
    elif dataset_name == IMAGE_REGULAR_TYPES[4]:
        return ShoesSettings().json_to_object(params)