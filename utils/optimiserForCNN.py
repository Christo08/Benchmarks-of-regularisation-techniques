import math

import pyhopper
import torch
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from NNs.Images.basicCNN import Net
from utils.customDataset import CustomDataset
from utils.dataLoader import clean_labels, loadImagesDatasSet
from utils.lossFucntions import CustomCrossEntropyLoss


def train(params):
    pool_size = params['pool_size']
    out_channels = params['out_channels']
    kernel_size = params['kernel_size']
    kernel_stride = params['kernel_stride']
    for counter in range(1, params['number_of_convolutional_layers']):
        if pool_size[counter] > out_channels[counter-1]:
            pool_size[counter] = out_channels[counter-1]
        if kernel_size[counter] > pool_size[counter-1]:
            kernel_size[counter] = pool_size[counter-1]
        if kernel_stride[counter] > kernel_size[counter]:
            kernel_stride[counter] = kernel_size[counter]
    params['pool_size'] = pool_size
    params['out_channels'] = out_channels
    params['kernel_size'] = kernel_size
    params['kernel_stride'] = kernel_stride
    labels = training_set[1]
    features_tensor = training_set[0]
    number_of_outputs = len(labels.unique().tolist())
    labels_tensor = clean_labels(labels, number_of_outputs)

    if torch.cuda.is_available():
        features_tensor = features_tensor.cuda()
        labels_tensor = labels_tensor.cuda()

    kf = KFold(n_splits=settings.number_of_fold, shuffle=True)

    loss_function = CustomCrossEntropyLoss()
    loss = 0
    for fold, (train_index, test_index) in enumerate(kf.split(features_tensor)):
        # Initialize the network, optimizer, and loss function
        x_training, x_testing = features_tensor[train_index], features_tensor[test_index]
        y_training, y_testing = labels_tensor[train_index], labels_tensor[test_index]

        train_dataset = CustomDataset(x_training, y_training)

        train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)

        network = Net(in_channels=settings.in_channels,
                      number_of_convolutional_layers=params["number_of_convolutional_layers"],
                      out_channels=params["out_channels"],
                      kernel_size=params["kernel_size"],
                      kernel_stride=params["kernel_stride"],
                      pool_size=params["pool_size"],
                      pool_type=params["pool_type"],
                      number_of_hidden_layers=params["number_of_hidden_layers"],
                      number_of_neurons_in_layers=params["number_of_neurons_in_layers"],
                      output_size=number_of_outputs)
        if torch.cuda.is_available():
            network = network.cuda()

        optimizer = optim.SGD(network.parameters(), lr=params["learning_rate"], momentum=params["momentum"])

        for epoch in range(params["number_of_epochs"]):
            for batch in train_loader:
                network.train()
                training_outputs = network(batch['data'])
                training_loss = loss_function(training_outputs, batch['label'])

                optimizer.zero_grad()
                training_loss.backward()
                optimizer.step()
        loss += loss_function(network(x_testing), y_testing).item()

    return loss/settings.number_of_fold


datasets = ["Balls", "BeanLeafs", "Cifar10", "MNIST", "Shoes"]
names = "0.\t exit\n"
counter = 1
for dataset in datasets:
    names += str(counter) + ".\t " + dataset + "\n"
    counter += 1
names += str(counter) + ".\t All\n"

while True:
    nameIndex = int(input("Please select a dataset's name by enter a number:\n" + names))
    if nameIndex == 0:
        break
    datasetNames = []
    if nameIndex == 6:
        datasetNames = datasets
    else:
        datasetNames.append(datasets[nameIndex - 1])
    for dataset in datasetNames:
        training_set, validation_set, settings = loadImagesDatasSet(dataset)
        search = pyhopper.Search({
            "batch_size": pyhopper.int(16, 1024, power_of=2),
            "learning_rate": pyhopper.float(0.0005, 0.25, log=True),
            "momentum": pyhopper.float(0.0005, 0.25, log=True),
            "number_of_epochs": pyhopper.int(50, 500, multiple_of=50),
            "number_of_convolutional_layers": pyhopper.int(2, 7),
            "out_channels": pyhopper.int(2, 64, power_of=2, shape=7),
            "kernel_size": pyhopper.int(2, 64, power_of=2, shape=7),
            "kernel_stride": pyhopper.int(2, 64, power_of=2, shape=7),
            "pool_size": pyhopper.int(2, 64, power_of=2, shape=7),
            "pool_type": pyhopper.int(0, 1, shape=7),
            "number_of_hidden_layers": pyhopper.int(2, 7),
            "number_of_neurons_in_layers": pyhopper.int(50, 500, multiple_of=25, shape=7)
        })
        best_params = search.run(
            train,
            direction="min",
            steps=150,
            n_jobs="per-gpu",
        )

        test_acc = train(best_params)
        print(f"Tuned params test {dataset} accuracy: {test_acc:0.2f}%")
        print(dataset + ": ", best_params)
