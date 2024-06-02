import math

import pyhopper
import torch
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from NNs.Numeric.basicNN import Net
from utils.customDataset import CustomDataset
from utils.dataLoader import loadNumericDataSet, clean_labels
from utils.lossFucntions import CustomCrossEntropyLoss
from utils.lossFucntions import CustomCrossEntropyRegularisationTermLoss


def train(params):
    labels = training_set[1]
    features = training_set[0]
    number_of_outputs = len(labels.unique().tolist())
    labels_tensor = clean_labels(labels, number_of_outputs)

    features_tensor = torch.tensor(features.values, dtype=torch.float32)

    if torch.cuda.is_available():
        features_tensor = features_tensor.cuda()
        labels_tensor = labels_tensor.cuda()

    number_of_inputs = features_tensor.shape[1]

    kf = KFold(n_splits=setting.number_of_fold, shuffle=True)
    testing_loss = 0

    for fold, (train_index, test_index) in enumerate(kf.split(features_tensor)):
        x_training, x_testing = features_tensor[train_index], features_tensor[test_index]
        y_training, y_testing = labels_tensor[train_index], labels_tensor[test_index]

        train_dataset = CustomDataset(x_training, y_training)

        train_loader = DataLoader(train_dataset, batch_size=setting.batch_size, shuffle=True)

        network = Net(input_size=number_of_inputs,
                      hidden_sizes=setting.number_of_neurons_in_layers,
                      number_of_hidden_layers=setting.number_of_hidden_layers,
                      output_size=number_of_outputs)
        if torch.cuda.is_available():
            network = network.cuda()

        loss_function = CustomCrossEntropyRegularisationTermLoss(params["weight_decay"])
        optimizer = optim.SGD(network.parameters(), lr=setting.learning_rate, momentum=setting.momentum)

        for epoch in range(setting.number_of_epochs):
            for batch in train_loader:
                network.train()
                training_outputs = network(batch['data'])
                training_loss = loss_function(training_outputs, batch['label'], network)

                optimizer.zero_grad()
                training_loss.backward()
                optimizer.step()
        loss_function = CustomCrossEntropyLoss()
        testing_loss += loss_function(network(x_testing), y_testing).item()
    return testing_loss / setting.number_of_fold


datasets = ["Diabetes",
            "LiverCirrhosis",
            "Magic",
            "MfeatPixel",
            "RainInAustralia",
            "WhiteWineQuality"]
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
    if nameIndex == 7:
        datasetNames = datasets
    else:
        datasetNames.append(datasets[nameIndex - 1])
    for dataset in datasetNames:
        training_set, validation_set, setting = loadNumericDataSet(dataset)
        search = pyhopper.Search({
            "weight_decay": pyhopper.float(0.0001, 0.999)
        })
        best_params = search.run(
            train,
            direction="min",
            steps=150,
            n_jobs="per-gpu",
         )

        test_loss = train(best_params)
        print(f"Tuned params test {dataset} loss: {test_loss:0.2f}%")
        print(dataset + ": ", best_params)
