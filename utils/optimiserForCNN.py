import pyhopper
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from datetime import datetime

from NNs.Images.basicCNN import Net
from utils.CustomDataset import CustomDataset
from utils.dataLoader import loadImagesDatasSet


def train(params):
    labels = trainSet[1]
    features = trainSet[0]

    pool_size = []

    params["date_time"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    for counter in range(setting.number_of_convolutional_layers):
        if setting.out_channels[counter] < setting.pool_size[counter]:
            setting.pool_size[counter] = setting.out_channels[counter]

        pool_size.append((setting.pool_size[counter], setting.pool_size[counter]))

        if setting.pool_size[counter - 1] < setting.kernel_size[counter] and counter != 0:
            setting.kernel_size[counter] = setting.pool_size[counter - 1]
        elif setting.in_channels < setting.kernel_size[counter] and counter == 0:
            setting.kernel_size[counter] = setting.in_channels

        if setting.kernel_size[counter] < setting.kernel_stride[counter]:
            setting.kernel_stride[counter] = setting.kernel_size[counter]

    file = open("C:\\Users\\User\\OneDrive\\tuks\\master\\code\\logs.json", "a")
    file.write(str(params) + ",")
    file.write("\n")
    file.close()

    testing_accuracy = 0

    kf = KFold(n_splits=setting.number_of_fold, shuffle=True)

    for train_index, test_index in kf.split(features):
        x_training, x_testing = features[train_index], features[test_index]
        y_training, y_testing = labels[train_index], labels[test_index]

        network = Net(setting.in_channels,
                      setting.number_of_convolutional_layers,
                      setting.out_channels,
                      setting.kernel_size,
                      setting.kernel_stride,
                      pool_size,
                      setting.pool_type,
                      setting.number_of_hidden_layers,
                      setting.number_of_neurons_in_layers,
                      setting.output_size
                      )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            network = network.cuda(device=0)
            x_training = x_training.cuda(device=0)
            y_training = y_training.cuda(device=0)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(network.parameters(), lr=setting.learning_rate, momentum=setting.momentum, weight_decay= params["weight_decay"])

        train_dataset = CustomDataset(x_training, y_training)

        train_loader = DataLoader(train_dataset, batch_size=setting.batch_size, shuffle=True)

        for epoch in range(setting.number_of_epochs):
            for batch in train_loader:
                data = batch['data']
                label = batch['label']
                data = data.float()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    network = network.cuda(device=0)
                    data = data.cuda(device=0)
                    label = label.cuda(device=0)
                network.train()
                training_outputs = network(data)
                training_loss = criterion(training_outputs, label)

                optimizer.zero_grad()
                training_loss.backward()
                optimizer.step()

        if torch.cuda.is_available():
            x_testing = x_testing.float().cuda(device=0)
            y_testing = y_testing.cuda(device=0)
        testing_output = network(x_testing)

        _, predicted_labels = torch.max(testing_output, 1)
        correct_predictions = (predicted_labels == y_testing).sum().item()
        total_samples = y_testing.shape[0]
        testing_accuracy += correct_predictions / total_samples * 100

    return testing_accuracy / setting.number_of_fold


datasets = ["Balls",
            "BeanLeafs",
            # "Cifar10",
            # "MNIST",
            "Shoes"]
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
    if nameIndex == 4:
        datasetNames = datasets
    else:
        datasetNames.append(datasets[nameIndex - 1])
    for dataset in datasetNames:
        trainSet, validationSet, setting = loadImagesDatasSet(dataset, False)
        search = pyhopper.Search({
            "weight_decay": pyhopper.float(0, 0.99),
        })
        best_params = search.run(
            train,
            direction="max",
            runtime="12h",
            n_jobs="per-gpu",
        )
        test_acc = train(best_params)
        print(f"Tuned params test {dataset} accuracy: {test_acc:0.2f}%")
        print(dataset + ": ", best_params)
