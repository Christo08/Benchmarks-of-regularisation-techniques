import math

import pyhopper
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from NNs.Numeric.basicNN import Net
from utils.CustomDataset import CustomDataset
from utils.dataLoader import loadNumericDataSet


def train(params):
    labels = trainSet[1]
    features = trainSet[0]

    classes = labels.unique().tolist()

    features_tensor = torch.tensor(features.values, dtype=torch.float32)  # Assuming features are floats
    labels_tensor = torch.tensor(labels.values, dtype=torch.long)

    if torch.cuda.is_available():
        features_tensor = features_tensor.cuda()
        labels_tensor = labels_tensor.cuda()

    numberOfClasses = len(classes)

    numberOfInputs = features_tensor.shape[1]

    testing_accuracy = 0

    kf = KFold(n_splits=setting.number_of_fold, shuffle=True)

    for fold, (train_index, test_index) in enumerate(kf.split(features_tensor)):
        x_train, x_test = features_tensor[train_index], features_tensor[test_index]
        y_train, y_test = labels_tensor[train_index], labels_tensor[test_index]

        train_dataset = CustomDataset(x_train, y_train)

        train_loader = DataLoader(train_dataset, batch_size=setting.batch_size, shuffle=True)

        network = Net(input_size=numberOfInputs,
                      hidden_sizes=setting.number_of_neurons_in_layers,
                      number_of_hidden_layers=setting.number_of_hidden_layers,
                      output_size=numberOfClasses)
        if torch.cuda.is_available():
            network = network.cuda()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(network.parameters(), lr=setting.learning_rate, momentum=setting.momentum)

        for epoch in range(setting.number_of_epochs):
            network.train()
            for batch in train_loader:
                training_outputs = network(batch['data'])
                training_loss = criterion(training_outputs, batch['label'])

                optimizer.zero_grad()
                training_loss.backward()
                optimizer.step()
            if epoch % params["weight_perturbation_epoch_interval"] == 0 and epoch != 0:
                network.perturb_weights(perturbation_factor=params["weight_perturbation_amount"])

        network.eval()

        test_outputs = network(x_test)

        _, predicted_labels = torch.max(test_outputs, 1)
        correct_predictions = (predicted_labels == y_test).sum().item()
        total_samples = len(y_test)
        testing_accuracy += correct_predictions / total_samples * 100
    return testing_accuracy / setting.number_of_fold


datasets = ["Diabetes",
            "Magic",
            "MfeatPixel",
            "RainInAustralia",
            "WhiteWine"]
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
        trainSet, validationSet, setting = loadNumericDataSet(dataset)
        search = pyhopper.Search({
             "weight_perturbation_amount": pyhopper.float(0.001, 0.999),
             "weight_perturbation_epoch_interval": pyhopper.int(2, math.floor(setting.number_of_epochs / 4))
        })

        if dataset == "RainInAustralia":
            best_params = search.run(
                train,
                direction="max",
                runtime="12h",
                n_jobs="per-gpu",
            )
        else:
            best_params = search.run(
                train,
                direction="max",
                runtime="4h",
                n_jobs="per-gpu",
            )


        test_acc = train(best_params)
        print(f"Tuned params test {dataset} accuracy: {test_acc:0.2f}%")
        print(dataset + ": ", best_params)
