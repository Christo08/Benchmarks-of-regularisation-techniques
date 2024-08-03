import random
import time

import torch
from sklearn.model_selection import KFold
from torch import optim
from torch.utils.data import DataLoader

from NNs.Images.basicCNN import Net
from utils.customDataset import CustomDataset
from utils.dataLoader import clean_labels
from utils.lossFucntions import CustomCrossEntropyRegularisationTermLoss
from utils.monitor import Monitor


def run(dataset_name, settings, training_set, validation_set):
    print(dataset_name + " regularisation term run")
    seed = random.randint(1, 100000)
    print("Random Seed: ", seed)

    torch.manual_seed(seed)

    labels = training_set[1].float()
    features_tensor = training_set[0].float()

    number_of_outputs = len(labels.unique().tolist())
    labels_tensor = clean_labels(labels, number_of_outputs)

    y_validation = clean_labels(validation_set[1].float(), number_of_outputs)
    x_validation = validation_set[0].float()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        features_tensor = features_tensor.cuda()
        labels_tensor = labels_tensor.cuda()

        x_validation = x_validation.cuda()
        y_validation = y_validation.cuda()

    kf = KFold(n_splits=settings.number_of_fold, shuffle=True, random_state=seed)

    loss_function = CustomCrossEntropyRegularisationTermLoss(settings.weight_decay)
    monitor = Monitor("Regularisation term", dataset_name, seed, loss_function, settings.log_interval, x_validation, y_validation)

    start_time = time.time()

    for fold, (train_index, test_index) in enumerate(kf.split(features_tensor)):
        # Initialize the network, optimizer, and loss function
        x_training, x_testing = features_tensor[train_index], features_tensor[test_index]
        y_training, y_testing = labels_tensor[train_index], labels_tensor[test_index]

        train_dataset = CustomDataset(x_training, y_training)

        train_loader = DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True)

        network = Net(in_channels=settings.in_channels,
                      input_image_size=settings.image_size,
                      number_of_convolutional_layers=settings.number_of_convolutional_layers,
                      out_channels=settings.out_channels,
                      padding=settings.padding,
                      kernel_size=settings.kernel_size,
                      kernel_stride=settings.kernel_stride,
                      pool_size=settings.pool_size,
                      pool_type=settings.pool_type,
                      number_of_hidden_layers=settings.number_of_hidden_layers,
                      number_of_neurons_in_layers=settings.number_of_neurons_in_layers,
                      output_size=number_of_outputs)
        if torch.cuda.is_available():
            network = network.cuda()

        optimizer = optim.SGD(network.parameters(), lr=settings.learning_rate, momentum=settings.momentum)
        monitor.set_dataset(x_training, y_training, x_testing, y_testing, fold)

        for epoch in range(settings.number_of_epochs):
            for batch in train_loader:
                network.train()
                training_outputs = network(batch['data'])
                training_loss = loss_function(training_outputs, batch['label'], network)

                optimizer.zero_grad()
                training_loss.backward()
                optimizer.step()
                del training_outputs
                del batch
                torch.cuda.empty_cache()
            monitor.evaluate(network, epoch)
            torch.cuda.empty_cache()
        del network
        del x_training
        del y_training
        del x_testing
        del y_testing
        torch.cuda.empty_cache()
    end_time = time.time()
    print('Finished Training')

    return monitor.log_performance(start_time, end_time, settings)