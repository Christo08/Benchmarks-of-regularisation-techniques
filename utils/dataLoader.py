import json
from collections import defaultdict
import random

import numpy as np
import pandas as pd
import torch
import torchvision
from pandas import Series

from utils.settings import MNISTSettings, BallsSettings, BeanLeafSettings, ShoesSettings, CifarSettings, \
    MagicSettings, WhiteWineQualitySettings, MfeatPixelSettings, RainInAustraliaSettings, DiabetesSettings, \
    LiverCirrhosisSettings, FashionMNISTSettings


def loadNumericDataSet(dataset_name):
    if dataset_name == 'Diabetes':
        setting = DiabetesSettings()
    elif dataset_name == 'LiverCirrhosis':
        setting = LiverCirrhosisSettings()
    elif dataset_name == 'Magic':
        setting = MagicSettings()
    elif dataset_name == 'MfeatPixel':
        setting = MfeatPixelSettings()
    elif dataset_name == 'RainInAustralia':
        setting = RainInAustraliaSettings()
    else:
        setting = WhiteWineQualitySettings()
    with open('Data/validationSet.json', 'r') as file:
        validationSetJson = json.load(file)
    dataset = pd.read_csv(setting.path_to_data)
    labels = dataset.get('target')
    features = dataset.drop('target', axis=1)
    if not validationSetJson[dataset_name]["validation"]:
        seed = random.randint(1, 100000)
        validationSetJson[dataset_name]["validation"], validationSetJson[dataset_name][
            "train"] = createValidationSetIndex(labels, seed)
        with open('Data/validationSet.json', 'w') as json_file:
            json.dump(validationSetJson, json_file, indent=4)
    validation_set = (
        features.iloc[validationSetJson[dataset_name]["validation"]],
        labels[validationSetJson[dataset_name]["validation"]])
    training_set = (features.iloc[validationSetJson[dataset_name]["train"]], labels[validationSetJson[dataset_name]["train"]])

    return training_set, validation_set, setting


def loadImagesDatasSet(dataset_name, needGeometricTransformation):
    if dataset_name == 'Balls':
        setting = BallsSettings()
    elif dataset_name == 'BeanLeafs':
        setting = BeanLeafSettings()
    elif dataset_name == 'FashionMNIST':
        setting = FashionMNISTSettings()
    elif dataset_name == 'Cifar10':
        setting = CifarSettings()
    elif dataset_name == 'MNIST':
        setting = MNISTSettings()
    else:
        setting = ShoesSettings()

    if needGeometricTransformation and dataset_name != 'MNIST':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(setting.image_size),  # Resize the image
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(setting.mean, setting.std),
            torchvision.transforms.RandomRotation(degrees=(-180, 180))
        ])
    else:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(setting.image_size),  # Resize the image
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(setting.mean, setting.std)
        ])

    if dataset_name == 'Cifar10':
        dataset = torchvision.datasets.CIFAR10(root=setting.path_to_data, download=True, transform=transform)
    elif dataset_name == 'MNIST':
        dataset = torchvision.datasets.MNIST(root=setting.path_to_data, download=True, transform=transform)
    elif dataset_name == 'FashionMNIST':
        dataset = torchvision.datasets.FashionMNIST(root=setting.path_to_data, download=True, transform=transform)
    else:
        dataset = torchvision.datasets.ImageFolder(root=setting.path_to_data, transform=transform)
    with open('Data/validationSet.json', 'r') as file:
        validationSetJson = json.load(file)

    if dataset_name == 'MNIST':
        data, labels = dataset.data.clone().detach(), dataset.targets.clone().detach()
        features = data.unsqueeze(1)
    elif dataset_name == "Cifar10":
        data, labels = torch.tensor(dataset.data), torch.tensor(dataset.targets)
        features = data.permute(0, 3, 1, 2)
    elif dataset_name == 'FashionMNIST':
        data, labels = dataset.data.clone().detach(), dataset.targets.clone().detach()
        features = data.unsqueeze(1)
    else:
        dataArray = []
        labelsArray = []
        for datapoint, label in dataset:
            dataArray.append(datapoint)
            labelsArray.append(label)
        features = torch.stack(dataArray)
        labels = torch.tensor(labelsArray)

    if not validationSetJson[dataset_name]["validation"]:
        seed = random.randint(1, 100000)
        validationSetJson[dataset_name]["validation"], validationSetJson[dataset_name][
            "train"] = createValidationSetIndex(labels.tolist(), seed)
        with open('Data/validationSet.json', 'w') as json_file:
            json.dump(validationSetJson, json_file, indent=4)

    validationIndex = torch.tensor(validationSetJson[dataset_name]["validation"])
    validation_set = (torch.index_select(features, 0, validationIndex), torch.index_select(labels, 0, validationIndex))

    trainIndex = torch.tensor(validationSetJson[dataset_name]["train"])
    training_set = (torch.index_select(features, 0, trainIndex), torch.index_select(labels, 0, trainIndex))

    return training_set, validation_set, setting


def createValidationSetIndex(labels, random_seed, validation_percent=0.1):
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


def clean_labels(targets, num_classes):
    if isinstance(targets, Series):
        targets = torch.tensor(targets.values)
    targets = targets.to(torch.int64)
    num_samples = targets.size(0)

    output_tensor = torch.zeros(num_samples, num_classes)
    output_tensor.scatter_(1, targets.unsqueeze(1), 1)
    output_tensor = output_tensor.to(torch.float)
    return output_tensor
