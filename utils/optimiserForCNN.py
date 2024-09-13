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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels = training_set[1].float()
    features_tensor = training_set[0].float()

    number_of_outputs = len(labels.unique().tolist())
    labels_tensor = clean_labels(labels, number_of_outputs)

    kf = KFold(n_splits=settings.number_of_fold, shuffle=True)

    loss_function = CustomCrossEntropyLoss()
    loss = 0

    padding_sizes = []
    stride_sizes = []
    kernel_sizes = []

    pool_kernel_sizes = []
    pool_stride_sizes = []
    pool_padding_sizes = []

    input_size = settings.in_channels

    input_size, padding_size, stride_size, kernel_size = calculate_valid_padding_stride(input_size,
                                                                                        params["kernel_sizes"][0],
                                                                                        params["kernel_strides"][0],
                                                                                        params["padding_sizes"][0])
    padding_sizes.append(padding_size)
    kernel_sizes.append(kernel_size)
    stride_sizes.append(stride_size)

    input_size, pool_padding_size, pool_stride_size, pool_kernel_size = calculate_valid_padding_stride(input_size,
                                                                                                       params["pool_kernel_sizes"][0],
                                                                                                       params["pool_stride_sizes"][0],
                                                                                                       params["pool_padding_sizes"][0])
    pool_kernel_sizes.append(pool_kernel_size)
    pool_stride_sizes.append(pool_stride_size)
    pool_padding_sizes.append(pool_padding_size)

    for counter in range(1, params["number_of_convolutional_layers"]):
        print(counter)
        input_size, padding_size, stride_size, kernel_size = calculate_valid_padding_stride(input_size,
                                                                                            params["kernel_sizes"][counter],
                                                                                            params["kernel_strides"][counter],
                                                                                            params["padding_sizes"][counter])
        padding_sizes.append(padding_size)
        kernel_sizes.append(kernel_size)
        stride_sizes.append(stride_size)

        input_size, pool_padding_size, pool_stride_size, pool_kernel_size = calculate_valid_padding_stride(input_size,
                                                                                                           params["pool_kernel_sizes"][counter],
                                                                                                           params["pool_stride_sizes"][counter],
                                                                                                           params["pool_padding_sizes"][counter])
        pool_kernel_sizes.append(pool_kernel_size)
        pool_stride_sizes.append(pool_stride_size)
        pool_padding_sizes.append(pool_padding_size)


    params["padding_sizes"] = padding_sizes
    params["kernel_sizes"] = kernel_sizes
    params["stride_sizes"] = stride_sizes

    params["pool_kernel_sizes"] = pool_kernel_sizes
    params["pool_stride_sizes"] = pool_stride_sizes
    params["pool_padding_sizes"] = pool_padding_sizes

    for fold, (train_index, test_index) in enumerate(kf.split(features_tensor)):
        # Initialize the network, optimizer, and loss function
        x_training, x_testing = features_tensor[train_index], features_tensor[test_index]
        y_training, y_testing = labels_tensor[train_index], labels_tensor[test_index]

        train_dataset = CustomDataset(x_training, y_training)

        train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)

        network = Net(in_channels=settings.in_channels,
                      input_image_size=settings.image_size,
                      number_of_convolutional_layers=params["number_of_convolutional_layers"],
                      out_channels=params["out_channels"],
                      padding=params["padding_sizes"],
                      kernel_size=params["kernel_sizes"],
                      kernel_stride=params["kernel_sizes"],
                      pool_kernel_size=params["pool_kernel_sizes"],
                      pool_type=params["pool_type"],
                      number_of_hidden_layers=params["number_of_hidden_layers"],
                      number_of_neurons_in_layers=params["number_of_neurons_in_layers"],
                      output_size=number_of_outputs,
                      pool_padding=params["pool_padding_sizes"],
                      pool_kernel_stride=params["pool_stride_sizes"])
        network = network.to(device)

        optimizer = optim.SGD(network.parameters(), lr=params["learning_rate"], momentum=params["momentum"])

        for epoch in range(params["number_of_epochs"]):
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
            testing_outputs = network(x_testing)
            loss += loss_function(testing_outputs, y_testing)

    return loss / settings.number_of_fold


def calculate_valid_padding_stride(input_size, kernel_size, stride, padding):
    """
    Adjust padding and stride to avoid output dimension becoming zero.
    """
    output_size = ((input_size - kernel_size + 2 * padding) // stride) + 1
    while output_size <= 0 and kernel_size < 512:
        while output_size <= 0 and padding < kernel_size // 2:
            while output_size <= 0 and stride > 1:
                stride -= 1
                output_size = ((input_size - kernel_size + 2 * padding) // stride) + 1
                if output_size > 0 and stride > 1:
                    return output_size, padding, stride, kernel_size

            padding += 1
            stride = 10
            output_size = ((input_size - kernel_size + 2 * padding) // stride) + 1
            if output_size > 0 and padding < kernel_size // 2:
                return output_size, padding, stride, kernel_size

        kernel_size += kernel_size
        padding = 0
        output_size = ((input_size - kernel_size + 2 * padding) // stride) + 1

    return output_size, padding, stride, kernel_size


datasets = ["Balls", "BeanLeafs", "FashionMNIST", "Cifar10", "MNIST", "Shoes"]
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
        "batch_size": pyhopper.int(16, 128, power_of=2),
        "learning_rate": pyhopper.float(0.0005, 0.25, log=True),
        "momentum": pyhopper.float(0.0005, 0.25, log=True),
        "number_of_epochs": pyhopper.int(50, 500, multiple_of=50),
        "number_of_convolutional_layers": pyhopper.int(1, 10),
        "out_channels": pyhopper.int(32, 64, power_of=2, shape=10),

        "kernel_sizes": pyhopper.int(1, 10, shape=10),
        "kernel_strides": pyhopper.int(1, 10, shape=10),
        "padding_sizes": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

        "pool_kernel_sizes": pyhopper.int(1, 10, shape=10),
        "pool_stride_sizes": pyhopper.int(1, 10, shape=10),
        "pool_padding_sizes": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

        "pool_type": pyhopper.int(0, 1, shape=10),
        "number_of_hidden_layers": pyhopper.int(1, 5),
        "number_of_neurons_in_layers": pyhopper.int(50, 1000, multiple_of=50, shape=5),
    })
    best_params = search.run(
        train,
        direction="min",
        steps=150,
        n_jobs="per-gpu",
        checkpoint_path="C:\\Users\\User\\OneDrive\\tuks\\master\\code\\CheckPoints\\" + dataset + "Checkpoint"
    )

    test_acc = train(best_params)
    print(f"Tuned params test {dataset} loss: {test_acc:0.2f}")
    print(dataset + ": ", best_params)
