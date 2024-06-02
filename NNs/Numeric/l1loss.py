import random
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from NNs.Numeric.basicNN import Net
from utils.customDataset import CustomDataset
from utils.runLogger import create_numeric_run_object


def run(dataset_name, setting, training_set, validation_set):
    print(dataset_name+" l1 loss run")
    seed = random.randint(1, 100000)
    print("Random Seed: ", seed)

    torch.manual_seed(seed)

    labels = training_set[1]
    features = training_set[0]

    classes = labels.unique().tolist()

    features_tensor = torch.tensor(features.values, dtype=torch.float32)  # Assuming features are floats
    labels_tensor = torch.tensor(labels.values, dtype=torch.long)

    if torch.cuda.is_available():
        features_tensor = features_tensor.cuda()
        labels_tensor = labels_tensor.cuda()

    numberOfClasses = len(classes)

    number_of_inputs = features_tensor.shape[1]

    kf = KFold(n_splits=setting.number_of_fold, shuffle=True, random_state=seed)

    loss_function = CustomCrossEntropyLoss()
    monitor = Monitor("Layer normalisation", dataset_name, seed, loss_function, settings.log_interval, x_validation, y_validation)

    difference_in_accuracies = []

    x_validation = torch.tensor(validation_set[0].values, dtype=torch.float32)
    y_validation = torch.tensor(validation_set[1].values, dtype=torch.long)

    if torch.cuda.is_available():
        x_validation = x_validation.cuda()
        y_validation = y_validation.cuda()

    start_time = time.time()
    for fold, (train_index, test_index) in enumerate(kf.split(features_tensor)):
        x_training, x_testing = features_tensor[train_index], features_tensor[test_index]
        y_training, y_testing = labels_tensor[train_index], labels_tensor[test_index]

        train_dataset = CustomDataset(x_training, y_training)

        train_loader = DataLoader(train_dataset, batch_size=setting.batch_size, shuffle=True)

        network = Net(input_size=number_of_inputs,
                      hidden_sizes=setting.number_of_neurons_in_layers,
                      number_of_hidden_layers=setting.number_of_hidden_layers,
                      output_size=numberOfClasses)
        if torch.cuda.is_available():
            network=network.cuda()

        loss_function = nn.L1Loss()
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

            network.eval()

            test_outputs = network(x_testing)
            testing_loss = loss_function(test_outputs, y_testing)

            validation_outputs = network(x_validation)
            validation_loss = loss_function(validation_outputs, y_validation)

            _, predicted_labels = torch.max(training_outputs, 1)
            correct_predictions = (predicted_labels == y_training).sum().item()
            total_samples = len(y_training)
            training_accuracy = correct_predictions / total_samples * 100

            _, predicted_labels = torch.max(test_outputs, 1)
            correct_predictions = (predicted_labels == y_testing).sum().item()
            total_samples = len(y_testing)
            testing_accuracy = correct_predictions / total_samples * 100

            _, predicted_labels = torch.max(validation_outputs, 1)
            correct_predictions = (predicted_labels == y_validation).sum().item()
            total_samples = len(y_validation)
            validation_accuracy = correct_predictions / total_samples * 100

            if fold == 0:
                training_losses.append(training_loss.item())
                training_accuracies.append(training_accuracy)

                testing_losses.append(testing_loss.item())
                testing_accuracies.append(testing_accuracy)

                validation_losses.append(validation_loss.item())
                validation_accuracies.append(validation_accuracy)

                difference_in_accuracies.append(training_accuracy - testing_accuracy)
            else:
                training_losses[epoch] += training_loss.item()
                training_accuracies[epoch] += training_accuracy

                testing_losses[epoch] += testing_loss.item()
                testing_accuracies[epoch] += testing_accuracy

                validation_losses[epoch] += validation_loss.item()
                validation_accuracies[epoch] += validation_accuracy

                difference_in_accuracies[epoch] += (training_accuracy - testing_accuracy)

            if epoch % setting.log_interval == 0:
                print(
                    "Fold %s, Epoch %s, Training loss %s, Testing loss %s, Training accuracy %s, Testing accuracy %s" % (
                        fold, epoch, training_loss.item(), testing_loss.item(), training_accuracy, testing_accuracy))

    for epoch in range(len(training_accuracies)):
        training_losses[epoch] /= setting.number_of_fold
        training_accuracies[epoch] /= setting.number_of_fold

        testing_losses[epoch] /= setting.number_of_fold
        testing_accuracies[epoch] /= setting.number_of_fold

        validation_losses[epoch] /= setting.number_of_fold
        validation_accuracies[epoch] /= setting.number_of_fold

        difference_in_accuracies[epoch] /= setting.number_of_fold
    end_time = time.time()
    print('Finished Training')

    return create_numeric_run_object("Baseline", seed, start_time, end_time, setting, training_losses,
                                     training_accuracies, testing_losses, testing_accuracies, validation_losses,
                                     validation_accuracies, difference_in_accuracies)