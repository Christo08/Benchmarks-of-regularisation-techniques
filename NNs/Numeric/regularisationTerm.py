import random
import time

import torch
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from NNs.Numeric.basicNN import Net
from utils.customDataset import CustomDataset
from utils.dataLoader import clean_labels
from utils.lossFucntions import CustomCrossEntropyRegularisationTermLoss
from utils.monitor import Monitor


def run(dataset_name, settings, training_set, validation_set):
    print(dataset_name + " regularisation term run")
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
    number_of_outputs = len(labels.unique().tolist())

    kf = KFold(n_splits=settings.number_of_fold, shuffle=True, random_state=seed)

    loss_function = CustomCrossEntropyRegularisationTermLoss(settings.weight_decay)
    monitor = Monitor("Regularisation term", dataset_name, seed, loss_function, settings.log_interval, x_validation, y_validation)

    start_time = time.time()

    for fold, (train_index, test_index) in enumerate(kf.split(features_tensor)):
        x_training, x_testing = features_tensor[train_index], features_tensor[test_index]
        y_training, y_testing = labels_tensor[train_index], labels_tensor[test_index]

        train_dataset = CustomDataset(x_training, y_training)

        train_loader = DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True)

        network = Net(input_size=number_of_inputs,
                      hidden_sizes=settings.number_of_neurons_in_layers,
                      number_of_hidden_layers=settings.number_of_hidden_layers,
                      output_size=number_of_outputs)
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
            monitor.evaluate(x_training, y_training, x_testing, y_testing, network, epoch, fold)

    end_time = time.time()
    print('Finished Training')

    return monitor.log_performance(start_time, end_time, settings)
