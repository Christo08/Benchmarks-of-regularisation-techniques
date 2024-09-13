import random
import time

import torch
import torch.optim as optim
from imblearn.over_sampling import SMOTENC, SMOTE
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from NNs.Numeric.basicNN import Net
from utils.customDataset import CustomDataset
from utils.dataLoader import clean_labels
from utils.lossFucntions import CustomCrossEntropyLoss
from utils.monitor import Monitor


def run(dataset_name, settings, training_set, validation_set):
    print(dataset_name + " SMOTE run")
    seed = random.randint(1, 100000)
    print("Random Seed: ", seed)

    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels = training_set[1]
    features = training_set[0]

    number_of_outputs = len(labels.unique().tolist())

    features_tensor = torch.tensor(features.values, dtype=torch.float32)  # Assuming features are floats
    labels_tensor = torch.tensor(labels.values, dtype=torch.int64)

    x_validation = torch.tensor(validation_set[0].values, dtype=torch.float32)
    y_validation = clean_labels(validation_set[1], number_of_outputs)

    number_of_inputs = features_tensor.shape[1]

    kf = KFold(n_splits=settings.number_of_fold, shuffle=True, random_state=seed)

    loss_function = CustomCrossEntropyLoss()

    monitor = Monitor(method="SMOTE",
                      dataset_name=dataset_name,
                      seed=seed,
                      loss_function=loss_function,
                      log_interval=settings.log_interval,
                      x_validation=x_validation,
                      y_validation=y_validation)

    if settings.categorical_features == []:
        oversample = SMOTE(random_state=seed)
    else:
        oversample = SMOTENC(random_state=seed, categorical_features=settings.categorical_features)

    start_time = time.time()

    for fold, (train_index, test_index) in enumerate(kf.split(features_tensor)):
        x_training, x_testing = features_tensor[train_index], features_tensor[test_index]
        y_training, y_testing = labels_tensor[train_index], labels_tensor[test_index]
        x_training, y_training = oversample.fit_resample(x_training, y_training)
        x_training = torch.tensor(x_training)
        y_training = torch.tensor(y_training)
        y_training = clean_labels(y_training, number_of_outputs)
        y_testing = clean_labels(y_testing, number_of_outputs)

        train_dataset = CustomDataset(x_training, y_training)

        train_loader = DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True)

        network = Net(input_size=number_of_inputs,
                      hidden_sizes=settings.number_of_neurons_in_layers,
                      number_of_hidden_layers=settings.number_of_hidden_layers,
                      output_size=number_of_outputs)
        network = network.to(device)

        optimizer = optim.SGD(network.parameters(), lr=settings.learning_rate, momentum=settings.momentum)
        monitor.set_dataset(x_training, y_training, x_testing, y_testing, fold)

        for epoch in range(settings.number_of_epochs):
            for batch in train_loader:
                batch_data = batch['data'].to(device)
                batch_label = batch['label'].to(device)

                network.train()
                training_outputs = network(batch_data)
                training_loss = loss_function(training_outputs, batch_label)

                optimizer.zero_grad()
                training_loss.backward()
                optimizer.step()

                del batch_data, batch_label
                torch.cuda.empty_cache()
            monitor.evaluate(network, epoch)

    end_time = time.time()
    print('Finished Training')

    return monitor.log_performance(start_time, end_time, settings)