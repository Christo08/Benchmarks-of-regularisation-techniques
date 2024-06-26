import json
from datetime import datetime

from matplotlib import pyplot as plt

# from NNs.Images.batchNormalisation import run as BatchNormalisationRun
# from NNs.Images.dropout import run as DropoutRun
# from NNs.Images.geometricTransformation import run as GeometricTransformationRun
# from NNs.Images.layerNormalisation import run as LayerNormalisationRun
from NNs.Images.baseline import run as BaselineImagesRun
# from NNs.Images.pruning import run as PruningRun
# from NNs.Images.regularisationTerm import run as RegularisationTermRun
# from NNs.Images.weightNormalisation import run as WeightNormalisationRun
# from NNs.Images.weightPerturbation import run as weightPerturbationRun

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


def drawPlot(lines, labels, xLabel, yLabel, title):
    epochs = [i + 1 for i in range(len(lines[0]))]
    plt.figure(figsize=(8, 6))
    # Plotting the first line
    for index in range(len(lines)):
        plt.plot(epochs, lines[index], label=labels[index])

    # Adding labels and title
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)

    # Adding a legend
    plt.legend(loc='center right')

    # Display the chart
    plt.show()


# def runImages(dataset_name, dataset, setting):
#     training_accuracies = []
#     testing_accuracies = []
#     validation_accuracies = []
#     difference_in_accuracies = []
#     accuracies = []
#     labels = []
#     dataset_run = {
#         "dataset_name": dataset_name,
#         "datasetPath": "Images",
#         "runs": []
#     }
#     if "13" in regularisationMethods:
#         regularisationMethods.clear()
#         regularisationMethods.extend(str(i) for i in range(1, 13))
#     if "1" in regularisationMethods:
#         runObject = BatchNormalisationRun(dataset_name, dataset, setting)
#         dataset_run["runs"].append(runObject)
#         if "4" in graphTypes:
#             accuracies.append(runObject["results"]["training_accuracies"])
#             accuracies.append(runObject["results"]["testing_accuracies"])
#             labels.append("Training accuracies")
#             labels.append("Testing accuracies")
#         else:
#             training_accuracies.append(runObject["results"]["training_accuracies"])
#             testing_accuracies.append(runObject["results"]["testing_accuracies"])
#             difference_in_accuracies.append(runObject["results"]["difference_in_accuracies"])
#             validation_accuracies.append(runObject["results"]["validation_accuracies"])
#             labels.append("Batch Normalisation")
#     if "2" in regularisationMethods:
#         print("1")
#         # runObject = ComboRun(dataset_name, dataset, setting)
#         # dataset_run["runs"].append(runObject)
#         # if "1" in graphTypes:
#         #     training_accuracies.append(runObject["results"]["training_accuracies"])
#         # if "2" in graphTypes:
#         #     testing_accuracies.append(runObject["results"]["testing_accuracies"])
#         # if "3" in graphTypes:
#         #     difference_in_accuracies.append(runObject["results"]["difference_in_accuracies"])
#         # if "4" in graphTypes:
#         #     accuracies.append(runObject["results"]["training_accuracies"])
#         #     accuracies.append(runObject["results"]["testing_accuracies"])
#         #     labels.append("Training accuracies")
#         #     labels.append("Testing accuracies")
#         # else:
#         #     labels.append("Combo Activation\nFunction")
#     if "3" in regularisationMethods:
#         print("2")
#         # runObject = CrossEntropyRun(dataset_name, dataset, setting)
#         # dataset_run["runs"].append(runObject)
#         # if "1" in graphTypes:
#         #     training_accuracies.append(runObject["results"]["training_accuracies"])
#         # if "2" in graphTypes:
#         #     testing_accuracies.append(runObject["results"]["testing_accuracies"])
#         # if "3" in graphTypes:
#         #     difference_in_accuracies.append(runObject["results"]["difference_in_accuracies"])
#         # if "4" in graphTypes:
#         #     accuracies.append(runObject["results"]["training_accuracies"])
#         #     accuracies.append(runObject["results"]["testing_accuracies"])
#         #     labels.append("Training accuracies")
#         #     labels.append("Testing accuracies")
#         # else:
#         #     labels.append("Cross-Entropy Loss\nFunction")
#     if "4" in regularisationMethods:
#         runObject = DropoutRun(dataset_name, dataset, setting)
#         dataset_run["runs"].append(runObject)
#         if "4" in graphTypes:
#             accuracies.append(runObject["results"]["training_accuracies"])
#             accuracies.append(runObject["results"]["testing_accuracies"])
#             labels.append("Training accuracies")
#             labels.append("Testing accuracies")
#         else:
#             training_accuracies.append(runObject["results"]["training_accuracies"])
#             testing_accuracies.append(runObject["results"]["testing_accuracies"])
#             difference_in_accuracies.append(runObject["results"]["difference_in_accuracies"])
#             validation_accuracies.append(runObject["results"]["validation_accuracies"])
#             labels.append("Dropout")
#     if "5" in regularisationMethods:
#         contine = True
#         if dataset_name == 'Balls':
#             dataset, settings = loadBallsData(True)
#         elif dataset_name == "Bean Leaf":
#             dataset, settings = loadBeanLeafData(True)
#         elif dataset_name == "Cifar-10":
#             dataset, settings = loadCifarData(True)
#         elif dataset_name == "MNIST":
#             # dataset, settings = loadMNISTData(True)
#             contine = False
#         else:
#             dataset, settings = loadShoesData(True)
#
#         if contine:
#             runObject = GeometricTransformationRun(dataset_name, dataset, setting)
#             dataset_run["runs"].append(runObject)
#             if "4" in graphTypes:
#                 accuracies.append(runObject["results"]["training_accuracies"])
#                 accuracies.append(runObject["results"]["testing_accuracies"])
#                 labels.append("Training accuracies")
#                 labels.append("Testing accuracies")
#             else:
#                 training_accuracies.append(runObject["results"]["training_accuracies"])
#                 testing_accuracies.append(runObject["results"]["testing_accuracies"])
#                 difference_in_accuracies.append(runObject["results"]["difference_in_accuracies"])
#                 validation_accuracies.append(runObject["results"]["validation_accuracies"])
#                 labels.append("Geometric\nTransformation")
#     if "6" in regularisationMethods:
#         runObject = LayerNormalisationRun(dataset_name, dataset, setting)
#         dataset_run["runs"].append(runObject)
#         if "4" in graphTypes:
#             accuracies.append(runObject["results"]["training_accuracies"])
#             accuracies.append(runObject["results"]["testing_accuracies"])
#             labels.append("Training accuracies")
#             labels.append("Testing accuracies")
#         else:
#             training_accuracies.append(runObject["results"]["training_accuracies"])
#             testing_accuracies.append(runObject["results"]["testing_accuracies"])
#             difference_in_accuracies.append(runObject["results"]["difference_in_accuracies"])
#             validation_accuracies.append(runObject["results"]["validation_accuracies"])
#             labels.append("Layer Normalisation")
#     if "7" in regularisationMethods:
#         runObject = LeakyReLURun(dataset_name, dataset, setting)
#         dataset_run["runs"].append(runObject)
#         if "4" in graphTypes:
#             accuracies.append(runObject["results"]["training_accuracies"])
#             accuracies.append(runObject["results"]["testing_accuracies"])
#             labels.append("Training accuracies")
#             labels.append("Testing accuracies")
#         else:
#             training_accuracies.append(runObject["results"]["training_accuracies"])
#             testing_accuracies.append(runObject["results"]["testing_accuracies"])
#             difference_in_accuracies.append(runObject["results"]["difference_in_accuracies"])
#             validation_accuracies.append(runObject["results"]["validation_accuracies"])
#             labels.append("LeakyReLU Activation\nFunction")
#     if "8" in regularisationMethods:
#         runObject = PruningRun(dataset_name, dataset, setting)
#         dataset_run["runs"].append(runObject)
#         if "4" in graphTypes:
#             accuracies.append(runObject["results"]["training_accuracies"])
#             accuracies.append(runObject["results"]["testing_accuracies"])
#             labels.append("Training accuracies")
#             labels.append("Testing accuracies")
#         else:
#             training_accuracies.append(runObject["results"]["training_accuracies"])
#             testing_accuracies.append(runObject["results"]["testing_accuracies"])
#             difference_in_accuracies.append(runObject["results"]["difference_in_accuracies"])
#             validation_accuracies.append(runObject["results"]["validation_accuracies"])
#             labels.append("Pruning")
#     if "9" in regularisationMethods:
#         runObject = RegularisationTermRun(dataset_name, dataset, setting)
#         dataset_run["runs"].append(runObject)
#         if "4" in graphTypes:
#             accuracies.append(runObject["results"]["training_accuracies"])
#             accuracies.append(runObject["results"]["testing_accuracies"])
#             labels.append("Training accuracies")
#             labels.append("Testing accuracies")
#         else:
#             training_accuracies.append(runObject["results"]["training_accuracies"])
#             testing_accuracies.append(runObject["results"]["testing_accuracies"])
#             difference_in_accuracies.append(runObject["results"]["difference_in_accuracies"])
#             validation_accuracies.append(runObject["results"]["validation_accuracies"])
#             labels.append("Regularisation Term")
#     if "10" in regularisationMethods:
#         print("3")
#         # runObject = SigmoidRun(dataset_name, dataset, setting)
#         # dataset_run["runs"].append(runObject)
#         # if "4" in graphTypes:
#         #     accuracies.append(runObject["results"]["training_accuracies"])
#         #     accuracies.append(runObject["results"]["testing_accuracies"])
#         #     labels.append("Training accuracies")
#         #     labels.append("Testing accuracies")
#         # else:
#         #     training_accuracies.append(runObject["results"]["training_accuracies"])
#         #     testing_accuracies.append(runObject["results"]["testing_accuracies"])
#         #     difference_in_accuracies.append(runObject["results"]["difference_in_accuracies"])
#         #     validation_accuracies.append(runObject["results"]["validation_accuracies"])
#         #     labels.append("Sigmoid Activation\nFunction")
#     if "11" in regularisationMethods:
#         runObject = WeightNormalisationRun(dataset_name, dataset, setting)
#         dataset_run["runs"].append(runObject)
#         if "4" in graphTypes:
#             accuracies.append(runObject["results"]["training_accuracies"])
#             accuracies.append(runObject["results"]["testing_accuracies"])
#             labels.append("Training accuracies")
#             labels.append("Testing accuracies")
#         else:
#             training_accuracies.append(runObject["results"]["training_accuracies"])
#             testing_accuracies.append(runObject["results"]["testing_accuracies"])
#             difference_in_accuracies.append(runObject["results"]["difference_in_accuracies"])
#             validation_accuracies.append(runObject["results"]["validation_accuracies"])
#             labels.append("Weight Normalisation")
#     if "12" in regularisationMethods:
#         runObject = weightPerturbationRun(dataset_name, dataset, setting)
#         dataset_run["runs"].append(runObject)
#         if "4" in graphTypes:
#             accuracies.append(runObject["results"]["training_accuracies"])
#             accuracies.append(runObject["results"]["testing_accuracies"])
#             labels.append("Training accuracies")
#             labels.append("Testing accuracies")
#         else:
#             training_accuracies.append(runObject["results"]["training_accuracies"])
#             testing_accuracies.append(runObject["results"]["testing_accuracies"])
#             difference_in_accuracies.append(runObject["results"]["difference_in_accuracies"])
#             validation_accuracies.append(runObject["results"]["validation_accuracies"])
#             labels.append("Weight Perturbation")
#
#     if "1" in graphTypes:
#         drawPlot(training_accuracies,
#                  labels,
#                  'Epochs',
#                  'Training accuracies',
#                  dataset_name + ': Epochs vs Training accuracies')
#     if "2" in graphTypes:
#         drawPlot(testing_accuracies,
#                  labels,
#                  'Epochs',
#                  'Testing accuracies',
#                  dataset_name + ': Epochs vs Testing accuracies')
#     if "3" in graphTypes:
#         drawPlot(difference_in_accuracies,
#                  labels,
#                  'Epochs',
#                  'Difference',
#                  dataset_name + ': Epochs vs Difference in training and testing accuracies')
#     if "4" in graphTypes:
#         drawPlot(accuracies,
#                  labels,
#                  'Epochs',
#                  'Difference',
#                  dataset_name + ': Epochs vs accuracies')
#
#     return dataset_run
#

def saveRuns(file, datasetsRuns):
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
            dataset_name.append(datasets[int(domain)-1])

    return dataset_name


def selectsRegularisationMethodFunction(datasetType):
    if datasetType == "1":
        regularisationMethods = ["Baseline", "Batch Normalisation", "Dropout",  "Geometric Transformation",
                                 "Layer Normalisation", "Pruning", "Regularisation Term", "Weight Normalisation",
                                 "Weight Perturbation"]
    else:
        regularisationMethods = ["Baseline", "Batch Normalisation", "Dropout",  "Layer Normalisation",
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
        selectedRegularisationMethods.extend(str(i) for i in range(1, len(regularisationMethods)+1))

    runFunctions = []
    if selectDataType == "1":
        runTime = "ImagesRun"
    else:
        runTime = "NumericRun"
    for method in selectedRegularisationMethods:
        functionName = regularisationMethods[int(method)-1].replace(" ", "") + runTime
        runFunctions.append(globals()[functionName])
    return runFunctions


selectDataType = input("Select the data type: \n1. Images \n2. Numeric\n")

datasetNames = selectDataset(selectDataType)
regularisationMethodFunction = selectsRegularisationMethodFunction(selectDataType)

current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

for dataset_name in datasetNames:
    dataset_run = {
        "dataset_name": dataset_name,
        "datasetPath": "",
        "runs": []
    }
    if selectDataType == "1":
        filename = "Results//Images//" + formatted_datetime + "_" + dataset_name + '.json'
        training_set, validation_set, setting = loadImagesDatasSet(dataset_name)
    else:
        filename = "Results//Numeric//" + formatted_datetime + "_" + dataset_name + '.json'
        training_set, validation_set, setting = loadNumericDataSet(dataset_name)
    dataset_run["datasetPath"] = setting.path_to_data
    for runFunction in regularisationMethodFunction:
        runObject = runFunction(dataset_name, setting, training_set, validation_set)
        dataset_run["runs"].append(runObject)
        saveRuns(filename, dataset_run)