import math
import sys
sys.path.append('/mnt/lustre/users/copperman/Benchmarks-of-regularisation-techniques')

import pyhopper
import torch
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from NNs.Images.basicCNN import Net
from utils.customDataset import CustomDataset
from utils.dataLoader import clean_labels, loadImagesDatasSet
from utils.lossFucntions import CustomCrossEntropyLoss
from utils.lossFucntions import CustomCrossEntropyRegularisationTermLoss


def train(params):
    labels = training_set[1].float()
    features_tensor = training_set[0].float()
    number_of_outputs = len(labels.unique().tolist())
    labels_tensor = clean_labels(labels, number_of_outputs)

    if torch.cuda.is_available():
        features_tensor = features_tensor.cuda()
        labels_tensor = labels_tensor.cuda()

    kf = KFold(n_splits=settings.number_of_fold, shuffle=True)

    loss_function = CustomCrossEntropyRegularisationTermLoss(params["weight_decay"])
    loss = 0
    for fold, (train_index, test_index) in enumerate(kf.split(features_tensor)):
        # Initialize the network, optimizer, and loss function
        x_training, x_testing = features_tensor[train_index], features_tensor[test_index]
        y_training, y_testing = labels_tensor[train_index], labels_tensor[test_index]

        train_dataset = CustomDataset(x_training, y_training)

        train_loader = DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True)

        network = Net(in_channels=settings.in_channels,
                      number_of_convolutional_layers=settings.number_of_convolutional_layers,
                      out_channels=settings.out_channels,
                      kernel_size=settings.kernel_size,
                      kernel_stride=settings.kernel_stride,
                      pool_size=settings.pool_size,
                      pool_type=settings.pool_type,
                      number_of_hidden_layers=settings.number_of_hidden_layers,
                      number_of_neurons_in_layers=settings.number_of_neurons_in_layers,
                      output_size=number_of_outputs,
                      input_image_size=settings.image_size,
                      padding=settings.padding,)
        if torch.cuda.is_available():
            network = network.cuda()

        optimizer = optim.SGD(network.parameters(), lr=settings.learning_rate, momentum=settings.momentum)

        for epoch in range(settings.number_of_epochs):
            for batch in train_loader:
                network.train()
                training_outputs = network(batch['data'])
                training_loss = loss_function(training_outputs, batch['label'], network)

                optimizer.zero_grad()
                training_loss.backward()
                optimizer.step()
        loss += loss_function(network(x_testing), y_testing, network).item()

    return loss / settings.number_of_fold


datasets = ["Balls", "BeanLeafs", "Cifar10", "MNIST", "Shoes"]
names = "0.\t exit\n"
counter = 1
for dataset in datasets:
    names += str(counter) + ".\t " + dataset + "\n"
    counter += 1
names += str(counter) + ".\t All\n"

nameIndex = input("Please select a dataset's name by enter a number:\n" + names)

datasetNames = []
if nameIndex == "6":
    datasetNames = datasets
else:
    indexes = nameIndex.split(" ")
    for index in indexes:
        datasetNames.append(datasets[int(index) - 1])
for dataset in datasetNames:
    print(f"Starting {dataset}")
    training_set, validation_set, settings = loadImagesDatasSet(dataset, False)
    search = pyhopper.Search({
        "weight_decay": pyhopper.float(0.005, 0.9)
    })
    best_params = search.run(
        train,
        direction="min",
        steps=50,
        n_jobs="per-gpu",
        checkpoint_path="C:\\Users\\User\\OneDrive\\tuks\\master\\code\\CheckPoints\\"+dataset+"Checkpoint"
    )

    test_acc = train(best_params)
    print(f"Tuned params test {dataset} loss: {test_acc:0.2f}")
    print(dataset + ": ", best_params)
