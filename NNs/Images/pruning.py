import random
import time

import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch import optim
from torch.utils.data import DataLoader

from NNs.Images.basicCNN import Net
from utils.customDataset import CustomDataset
from utils.runLogger import create_numeric_run_object


def run(dataset_name, dataset, setting):
    print(dataset_name+" pruning")
    seed = random.randint(1, 100000)
    torch.manual_seed(seed)

    # Create KFold cross-validator
    kf = KFold(n_splits=setting.number_of_fold, shuffle=True, random_state=seed)
    fold = 1

    training_accuracies = []
    testing_accuracies = []
    difference_in_accuracies = []

    if dataset_name == 'MNIST':
        data, labels = torch.tensor(dataset.data),  torch.tensor(dataset.targets)
        data = data.unsqueeze(1)
    elif dataset_name == "Cifar-10":
        data, labels = torch.tensor(dataset.data),  torch.tensor(dataset.targets)
        data = data.permute(0, 3, 1, 2)
    else:
        dataArray =[]
        labelsArray = []
        for datapoint, label in dataset:
            dataArray.append(datapoint)
            labelsArray.append(label)
        data = torch.stack(dataArray)
        labels = torch.tensor(labelsArray)
    start_time = time.time()

    for train_indices, val_indices in kf.split(data):
        # Initialize the network, optimizer, and loss function
        x_training, y_training = data[train_indices], labels[train_indices]
        x_testing, y_testing = data[train_indices], labels[train_indices]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            x_testing = x_testing.cuda().float()
            x_training = x_training.cuda().float()
            y_testing = y_testing.cuda()
            y_training = y_training.cuda()

        train_dataset = CustomDataset(x_training, y_training)

        train_loader = DataLoader(train_dataset, batch_size=setting.batch_size, shuffle=True)

        network = Net(setting.in_channels, setting.out_channels, setting.kernel_size, setting.pool_size, setting.hidden_layer_sizes, setting.output_size, setting.stride)
        if torch.cuda.is_available():
            network=network.cuda()

        # Define loss function and optimizer
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.SGD(network.parameters(), lr=setting.learning_rate, momentum=setting.momentum)

        for epoch in range(setting.number_of_epochs):
            for batch in train_loader:
                network.train()
                training_outputs = network(batch['data'])
                training_loss = loss_function(training_outputs, batch['label'])

                optimizer.zero_grad()
                training_loss.backward()
                optimizer.step()

            training_outputs = network(x_training)
            training_loss = loss_function(training_outputs, y_training)

            testing_output = network(x_testing)
            testing_loss = loss_function(testing_output, y_testing)

            if epoch % setting.prune_epoch_interval == 0 and epoch != 0:
                network.prune(amount=setting.prune_amount)

            _, predicted_labels = torch.max(training_outputs, 1)
            correct_predictions = (predicted_labels == y_training).sum().item()
            total_samples = y_training.shape[0]
            training_accuracy = correct_predictions / total_samples * 100

            _, predicted_labels = torch.max(testing_output, 1)
            correct_predictions = (predicted_labels == y_testing).sum().item()
            total_samples = y_testing.shape[0]
            testing_accuracy = correct_predictions / total_samples * 100

            if fold == 1:
                training_accuracies.append(training_accuracy)
                testing_accuracies.append(testing_accuracy)
                difference_in_accuracies.append(training_accuracy - testing_accuracy)
            else:
                training_accuracies[epoch] += training_accuracy
                testing_accuracies[epoch] += testing_accuracy
                difference_in_accuracies[epoch] += (training_accuracy - testing_accuracy)

            if epoch % setting.log_interval == 0:
                print("Fold %s, Epoch %s, Training loss %s, Testing loss %s, Training accuracy %s, Testing accuracy %s" % (fold, epoch, training_loss.item(), testing_loss.item(),training_accuracy,testing_accuracy))

        fold +=1

    for epoch in range(len(training_accuracies)):
        training_accuracies[epoch] /= setting.number_of_fold
        testing_accuracies[epoch] /= setting.number_of_fold
        difference_in_accuracies[epoch] /= setting.number_of_fold
    end_time = time.time()
    print('Finished Training')

    return create_numeric_run_object("Pruning", seed, start_time, end_time, setting,
                                     training_accuracies, testing_accuracies, difference_in_accuracies)