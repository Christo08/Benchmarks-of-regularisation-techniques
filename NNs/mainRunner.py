import json
from datetime import datetime

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

from utils.dataLoader import load_numeric_data_set, load_images_datas_set
from utils.menus import (show_dataset_menu,
                         show_model_menu,
                         show_regularisation_menu,
                         MODULE_TYPES,
                         IMAGE_REGULAR_TYPES,
                         IMAGE_DATASETS,
                         NUMERIC_REGULAR_TYPES)

graphTypes = []
regularisationMethods = []


def save_runs(file, datasetsRuns):
    with open(file, "w") as json_file:
        json.dump(datasetsRuns, json_file)

def make_run():
    while True:
        module_types = show_model_menu()
        if len(module_types) == 0:
            break
        while True:
            current_datetime = datetime.now()
            formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
            regularisation_method = show_regularisation_menu("BOTH" if len(module_types) == len(MODULE_TYPES) else module_types[0])
            if len(regularisation_method) == 0:
                break
            datasets = show_dataset_menu("ALL" if len(module_types) == len(MODULE_TYPES) else module_types[0])
            if len(datasets) == 0:
                continue
            for dataset in datasets:
                if dataset in IMAGE_DATASETS:
                    image_runner(regularisation_method, dataset, formatted_datetime)
                else:
                    numeric_runner(regularisation_method, dataset, formatted_datetime)

def image_runner(regularisation_method, dataset, formatted_datetime):
    techniques = [x for x in regularisation_method if x in IMAGE_REGULAR_TYPES]
    file_name = "Results//Images//" + formatted_datetime + "_" + dataset + '.json'
    dataset_run = {
        "dataset_name": dataset,
        "datasetPath": "",
        "runs": []
    }
    for technique in techniques:
        print(f"Running {technique} on {dataset}")
        training_set, validation_set, setting = load_images_datas_set(dataset, technique == IMAGE_REGULAR_TYPES[3])
        dataset_run["datasetPath"] = setting.path_to_data
        if technique == IMAGE_REGULAR_TYPES[0]:
            dataset_run["runs"].append(baseline_images_run(dataset, setting, training_set, validation_set))
        elif technique == IMAGE_REGULAR_TYPES[1]:
            dataset_run["runs"].append(batch_normalisation_images_run(dataset, setting, training_set, validation_set))
        elif technique == IMAGE_REGULAR_TYPES[2]:
            dataset_run["runs"].append(dropout_images_run(dataset, setting, training_set, validation_set))
        elif technique == IMAGE_REGULAR_TYPES[3]:
            dataset_run["runs"].append(geometric_transformation_images_run(dataset, setting, training_set, validation_set))
        elif technique == IMAGE_REGULAR_TYPES[4]:
            dataset_run["runs"].append(layer_normalisation_images_run(dataset, setting, training_set, validation_set))
        elif technique == IMAGE_REGULAR_TYPES[5]:
            dataset_run["runs"].append(pruning_images_run(dataset, setting, training_set, validation_set))
        elif technique == IMAGE_REGULAR_TYPES[6]:
            dataset_run["runs"].append(regularisation_term_images_run(dataset, setting, training_set, validation_set))
        elif technique == IMAGE_REGULAR_TYPES[7]:
            dataset_run["runs"].append(weight_normalisation_images_run(dataset, setting, training_set, validation_set))
        else:
            dataset_run["runs"].append(weight_perturbation_images_run(dataset, setting, training_set, validation_set))
        save_runs(file_name, dataset_run)

def numeric_runner(regularisation_method, dataset, formatted_datetime):
    techniques = [x for x in regularisation_method if x in NUMERIC_REGULAR_TYPES]
    file_name = "Results//Numeric//" + formatted_datetime + "_" + dataset + '.json'
    dataset_run = {
        "dataset_name": dataset,
        "datasetPath": "",
        "runs": []
    }
    for technique in techniques:
        print(f"Running {technique} on {dataset}")
        training_set, validation_set, setting = load_numeric_data_set(dataset)
        dataset_run["datasetPath"] = setting.path_to_data
        if technique == NUMERIC_REGULAR_TYPES[0]:
            dataset_run["runs"].append(baseline_numeric_run(dataset, setting, training_set, validation_set))
        elif technique == NUMERIC_REGULAR_TYPES[1]:
            dataset_run["runs"].append(batch_normalisation_numeric_run(dataset, setting, training_set, validation_set))
        elif technique == NUMERIC_REGULAR_TYPES[2]:
            dataset_run["runs"].append(dropout_numeric_run(dataset, setting, training_set, validation_set))
        elif technique == NUMERIC_REGULAR_TYPES[3]:
            dataset_run["runs"].append(layer_normalisation_numeric_run(dataset, setting, training_set, validation_set))
        elif technique == NUMERIC_REGULAR_TYPES[4]:
            dataset_run["runs"].append(pruning_numeric_run(dataset, setting, training_set, validation_set))
        elif technique == NUMERIC_REGULAR_TYPES[5]:
            dataset_run["runs"].append(regularisation_term_numeric_run(dataset, setting, training_set, validation_set))
        elif technique == NUMERIC_REGULAR_TYPES[6]:
            dataset_run["runs"].append(SMOTE_numeric_run(dataset, setting, training_set, validation_set))
        elif technique == NUMERIC_REGULAR_TYPES[7]:
            dataset_run["runs"].append(weight_normalisation_numeric_run(dataset, setting, training_set, validation_set))
        else:
            dataset_run["runs"].append(weight_perturbation_numeric_run(dataset, setting, training_set, validation_set))
        save_runs(file_name, dataset_run)
