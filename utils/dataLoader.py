import json
from collections import defaultdict
import random

import numpy as np
import pandas as pd
import torch
import torchvision

from utils.settings import MNISTSettings, BallsSettings, BeanLeafSettings, ShoesSettings, CifarSettings, \
    MagicSettings, WhiteWineQualitySettings, MfeatPixelSettings, RainInAustraliaSettings, DiabetesSettings

def loadNumericDataSet(datasetName):
    if datasetName == 'Diabetes':
        setting = DiabetesSettings()
    elif datasetName == 'Magic':
        setting = MagicSettings()
    elif datasetName == 'MfeatPixel':
        setting = MfeatPixelSettings()
    elif datasetName == 'RainInAustralia':
        setting = RainInAustraliaSettings()
    else:
        setting = WhiteWineQualitySettings()
    with open('C:\\Users\\User\\OneDrive\\tuks\\master\\code\\Data\\validationSet.json', 'r') as file:
        validationSetJson = json.load(file)
    dataset = pd.read_csv(setting.path_to_data)
    labels = dataset.get('target')
    features = dataset.drop('target', axis=1)
    if not validationSetJson[datasetName]["validation"]:
        seed = random.randint(1, 100000)
        validationSetJson[datasetName]["validation"],validationSetJson[datasetName]["train"] = createValidationSetIndex(labels,seed)
        with open('C:\\Users\\User\\OneDrive\\tuks\\master\\code\\Data\\validationSet.json', 'w') as json_file:
            json.dump(validationSetJson, json_file, indent=4)
    validationSet = (features.iloc[validationSetJson[datasetName]["validation"]], labels[validationSetJson[datasetName]["validation"]])
    trainSet = (features.iloc[validationSetJson[datasetName]["train"]], labels[validationSetJson[datasetName]["train"]])

    return trainSet, validationSet, setting


def loadImagesDatasSet(datasetName, rotation):
    if datasetName == 'Balls':
        setting = BallsSettings()
    elif datasetName == 'BeanLeafs':
        setting = BeanLeafSettings()
    elif datasetName == 'Cifar10':
        setting = CifarSettings()
    elif datasetName == 'MNIST':
        setting = MNISTSettings()
    else:
        setting = ShoesSettings()

    if(rotation):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomRotation(setting.rotation),
            torchvision.transforms.Resize(setting.image_size),  # Resize the image
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(setting.mean, setting.std)
        ])
    else:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(setting.image_size),  # Resize the image
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(setting.mean, setting.std)
        ])
    if datasetName == 'Cifar10':
        dataset = torchvision.datasets.CIFAR10(root=setting.path_to_data, download=True, transform=transform)
    elif datasetName == 'MNIST':
        dataset = torchvision.datasets.MNIST(root=setting.path_to_data, download=True, transform=transform)
    else:
        dataset = torchvision.datasets.ImageFolder(root=setting.path_to_data, transform=transform)
    with open('C:\\Users\\User\\OneDrive\\tuks\\master\\code\\Data\\validationSet.json', 'r') as file:
        validationSetJson = json.load(file)

    if datasetName == 'MNIST':
        data, labels = torch.tensor(dataset.data),  torch.tensor(dataset.targets)
        features = data.unsqueeze(1)
    elif datasetName == "Cifar10":
        data, labels = torch.tensor(dataset.data),  torch.tensor(dataset.targets)
        features = data.permute(0, 3, 1, 2)
    else:
        dataArray =[]
        labelsArray = []
        for datapoint, label in dataset:
            dataArray.append(datapoint)
            labelsArray.append(label)
        features = torch.stack(dataArray)
        labels = torch.tensor(labelsArray)

    if not validationSetJson[datasetName]["validation"]:
        seed = random.randint(1, 100000)
        validationSetJson[datasetName]["validation"],validationSetJson[datasetName]["train"] = createValidationSetIndex(labels.tolist(),seed)
        with open('C:\\Users\\User\\OneDrive\\tuks\\master\\code\\Data\\validationSet.json', 'w') as json_file:
            json.dump(validationSetJson, json_file, indent=4)

    validationIndex = torch.tensor(validationSetJson[datasetName]["validation"])
    validationSet = (torch.index_select(features, 0, validationIndex), torch.index_select(labels, 0, validationIndex))

    trainIndex = torch.tensor(validationSetJson[datasetName]["train"])
    trainSet = (torch.index_select(features, 0, trainIndex), torch.index_select(labels, 0, trainIndex))

    return trainSet, validationSet, setting


def createValidationSetIndex(labels, random_seed, validation_percent = 0.1):
    class_distribution = defaultdict(list)
    for counter, label in enumerate(labels):
        class_distribution[label].append(counter)

    class_indices = list(class_distribution.values())

    for indices in class_indices:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    validation_indices, test_indices = [], []
    for indices in class_indices:
        num_validation = int(validation_percent * len(indices))
        test_indices.extend(indices[num_validation:])
        validation_indices.extend(indices[:num_validation])

    return validation_indices, test_indices