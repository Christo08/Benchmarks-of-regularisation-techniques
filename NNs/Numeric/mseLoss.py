import random
import time
from datetime import datetime

import torch
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from NNs.Numeric.basicNN import Net
from utils.customDataset import CustomDataset
from utils.dataLoader import clean_labels
from utils.lossFucntions import CustomCrossEntropyLoss, CustomMSELoss


def run(dataset_name, setting, training_set, validation_set):
    print(dataset_name + " MSE loss run")
    seed = random.randint(1, 100000)
    print("Random Seed: ", seed)

    torch.manual_seed(seed)

    labels = training_set[1]
    features = training_set[0]

    number_of_outputs = len(labels.unique().tolist())
    labels_tensor = clean_labels(labels, number_of_outputs)

    x_validation = torch.tensor(validation_set[0].values, dtype=torch.float32)
    y_validation = clean_labels(validation_set[1], number_of_outputs)

    features_tensor = torch.tensor(features.values, dtype=torch.float32)

    if torch.cuda.is_available():
        features_tensor = features_tensor.cuda()
        labels_tensor = labels_tensor.cuda()
        x_validation = x_validation.cuda()
        y_validation = y_validation.cuda()

    number_of_inputs = features_tensor.shape[1]

    kf = KFold(n_splits=setting.number_of_fold, shuffle=True, random_state=seed)

    loss_function = CustomCrossEntropyLoss()
    monitor = Monitor("Layer normalisation", dataset_name, seed, loss_function, settings.log_interval, x_validation, y_validation)

    start_time = time.time()
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

        loss_function = CustomMSELoss()
        optimizer = optim.SGD(network.parameters(), lr=setting.learning_rate, momentum=setting.momentum)

        for epoch in range(setting.number_of_epochs):
            for batch in train_loader:
                network.train()
                training_outputs = network(batch['data'])
                training_loss = loss_function(training_outputs, batch['label'])

                optimizer.zero_grad()
                training_loss.backward()
                optimizer.step()

            training_loss, training_accuracy = evaluatePerformance(x_training, y_training, network, loss_function)
            testing_loss, testing_accuracy = evaluatePerformance(x_testing, y_testing, network, loss_function)
            validation_loss, validation_accuracy = evaluatePerformance(x_validation, y_validation, network, loss_function)

            if epoch == 0:
                training_losses.append([])
                training_losses[fold].append(training_loss.item())

                training_accuracies.append([])
                training_accuracies[fold].append(training_accuracy)

                testing_losses.append([])
                testing_losses[fold].append(testing_loss.item())

                testing_accuracies.append([])
                testing_accuracies[fold].append(testing_accuracy)

                validation_losses.append([])
                validation_losses[fold].append(validation_loss.item())

                validation_accuracies.append([])
                validation_accuracies[fold].append(validation_accuracy)
            else:
                training_losses[fold].append(training_loss.item())

                training_accuracies[fold].append(training_accuracy)

                testing_losses[fold].append(testing_loss.item())

                testing_accuracies[fold].append(testing_accuracy)

                validation_losses[fold].append(validation_loss.item())

                validation_accuracies[fold].append(validation_accuracy)

            if epoch % setting.log_interval == 0:
                print("Fold %s Epoch %s @ %s: " % (fold, epoch, datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
                print("+------------+--------------------+---------------------+")
                print("| Type       | Loss \t\t\t  | Accuracy \t\t\t|")
                print("+------------+--------------------+---------------------+")
                print("| Training   | %s | %s \t| " % (training_loss.item(), training_accuracy))
                print("+------------+--------------------+---------------------+")
                print("| Testing    | %s | %s \t| " % (testing_loss.item(), testing_accuracy))
                print("+------------+--------------------+---------------------+")
                print("| Validation | %s | %s \t| " % (validation_loss.item(), validation_accuracy))
                print("+------------+--------------------+---------------------+")

    end_time = time.time()
    print('Finished Training')

    return create_numeric_run_object("MSE Loss", seed, start_time, end_time, setting, training_losses,
                                     training_accuracies, testing_losses, testing_accuracies, validation_losses,
                                     validation_accuracies)
