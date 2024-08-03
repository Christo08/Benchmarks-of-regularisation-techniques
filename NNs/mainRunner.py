import json
from datetime import datetime

import torch
from matplotlib import pyplot as plt

from NNs.Images.baseline import run as BaselineImagesRun
from NNs.Images.batchNormalisation import run as BatchNormalisationImagesRun
from NNs.Images.dropout import run as DropoutImagesRun
from NNs.Images.geometricTransformation import run as GeometricTransformationImagesRun
from NNs.Images.layerNormalisation import run as LayerNormalisationImagesRun
from NNs.Images.pruning import run as PruningImagesRun
from NNs.Images.regularisationTerm import run as RegularisationTermImagesRun
from NNs.Images.weightNormalisation import run as WeightNormalisationImagesRun
from NNs.Images.weightPerturbation import run as WeightPerturbationImagesRun

from NNs.Numeric.baseline import run as BaselineNumericRun
from NNs.Numeric.batchNormalisation import run as BatchNormalisationNumericRun
from NNs.Numeric.dropout import run as DropoutNumericRun
# from NNs.Numeric.l1loss import run as L1LossNumericRun
from NNs.Numeric.layerNormalisation import run as LayerNormalisationNumericRun
# from NNs.Numeric.mseLoss import run as MSELossNumericRun
from NNs.Numeric.pruning import run as PruningNumericRun
from NNs.Numeric.regularisationTerm import run as RegularisationTermNumericRun
from NNs.Numeric.smote import run as SMOTENumericRun
from NNs.Numeric.weightNormalisation import run as WeightNormalisationNumericRun
from NNs.Numeric.weightPerturbation import run as WeightPerturbationNumericRun

from utils.dataLoader import loadNumericDataSet, loadImagesDatasSet

graphTypes = []
regularisationMethods = []


def save_runs(file, datasetsRuns):
    with open(file, "w") as json_file:
        json.dump(datasetsRuns, json_file)


def selectDataset(datasetType):
    if datasetType == "1":
        datasets = ["Balls", "BeanLeafs", "Cifar10", "MNIST", "Shoes"]
    else:
        datasets = ["Diabetes", "LiverCirrhosis", "Magic", "MfeatPixel", "WhiteWineQuality"]

    inputText = "Select one or more of the problem domain:\n"
    counter = 1
    for dataset in datasets:
        inputText += str(counter) + ".\t " + dataset + "\n"
        counter += 1
    inputText += str(counter) + ".\t All\n"

    selectsProblemDomains = input(inputText)
    selectsProblemDomains = selectsProblemDomains.split(" ")
    dataset_name = []
    if str(counter) in selectsProblemDomains:
        selectsProblemDomains.clear()
        selectsProblemDomains.extend(str(i) for i in range(0, len(datasets)))
        for domain in selectsProblemDomains:
            dataset_name.append(datasets[int(domain)])
    else:
        for domain in selectsProblemDomains:
            dataset_name.append(datasets[int(domain) - 1])

    return dataset_name


def selectsRegularisationMethodFunction(datasetType):
    if datasetType == "1":
        regularisationMethods = ["Baseline", "Batch Normalisation", "Dropout", "Geometric Transformation",
                                 "Layer Normalisation", "Pruning", "Regularisation Term", "Weight Normalisation",
                                 "Weight Perturbation"]
    else:
        regularisationMethods = ["Baseline", "Batch Normalisation", "Dropout", "Layer Normalisation",
                                 "Pruning", "Regularisation Term", "SMOTE", "Weight Normalisation",
                                 "Weight Perturbation"]
    inputText = "Select one or more of the regularisation methods:\n"
    counter = 1
    for dataset in regularisationMethods:
        inputText += str(counter) + ".\t " + dataset + "\n"
        counter += 1
    inputText += str(counter) + ".\t All\n"

    selectedRegularisationMethods = input(inputText)
    selectedRegularisationMethods = selectedRegularisationMethods.split(" ")
    if str(counter) in selectedRegularisationMethods:
        selectedRegularisationMethods.clear()
        selectedRegularisationMethods.extend(str(i) for i in range(1, len(regularisationMethods) + 1))

    runFunctions = []
    if selectDataType == "1":
        runType = "ImagesRun"
    else:
        runType = "NumericRun"
    for method in selectedRegularisationMethods:
        functionName = regularisationMethods[int(method) - 1].replace(" ", "") + runType
        runFunctions.append((functionName.replace(runType, ""), globals()[functionName]))
    return runFunctions


selectDataType = input("Select the data type: \n1. Images \n2. Numeric\n")

datasetNames = selectDataset(selectDataType)
regularisationMethodFunction = selectsRegularisationMethodFunction(selectDataType)

current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

for dataset_name in datasetNames:
    reloadNeed = False
    dataset_run = {
        "dataset_name": dataset_name,
        "datasetPath": "",
        "runs": []
    }
    if selectDataType == "1":
        filename = "Results//Images//" + formatted_datetime + "_" + dataset_name + '.json'
        training_set, validation_set, setting = loadImagesDatasSet(dataset_name, False)
    else:
        filename = "Results//Numeric//" + formatted_datetime + "_" + dataset_name + '.json'
        training_set, validation_set, setting = loadNumericDataSet(dataset_name)
    dataset_run["datasetPath"] = setting.path_to_data
    for runFunction in regularisationMethodFunction:
        if selectDataType == "1" and runFunction[0] == "GeometricTransformation":
            training_set, validation_set, setting = loadImagesDatasSet(dataset_name, True)
            reloadNeed = True
        elif selectDataType == "1" and reloadNeed:
            training_set, validation_set, setting = loadImagesDatasSet(dataset_name, False)
            reloadNeed = False

        runObject = runFunction[1](dataset_name, setting, training_set, validation_set)
        dataset_run["runs"].append(runObject)
        save_runs(filename, dataset_run)
