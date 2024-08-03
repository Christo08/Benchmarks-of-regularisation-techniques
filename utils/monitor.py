import math

import torch
from datetime import datetime

from torch.utils.data import DataLoader

from utils.customDataset import CustomDataset
from utils.lossFucntions import CustomCrossEntropyRegularisationTermLoss
from sklearn.metrics import f1_score


class Monitor:
    def __init__(self, method, dataset_name, seed, loss_function, log_interval, x_validation, y_validation):
        torch.cuda.empty_cache()
        self.fold = 0
        self.x_training = None
        self.y_training = None
        self.x_testing = None
        self.y_testing = None
        self.model = None
        self.loss_function = loss_function.to('cpu')
        self.log_interval = log_interval
        self.method = method
        self.seed = seed
        self.dataset_name = dataset_name

        self.training_losses = []
        self.training_accuracies = []
        self.training_f1_scores = []

        self.testing_losses = []
        self.testing_accuracies = []
        self.testing_f1_scores = []

        self.validation_losses = []
        self.validation_accuracies = []
        self.validation_f1_scores = []

        self.x_validation = x_validation.to('cpu')
        self.y_validation = y_validation.to('cpu')

    def evaluate(self, model, epoch):
        cpuModel = model.cpu()
        training_loss, training_accuracy, training_f1_score = self.evaluate_performance(cpuModel, self.x_training,
                                                                                        self.y_training)
        testing_loss, testing_accuracy, testing_f1_score = self.evaluate_performance(cpuModel, self.x_testing,
                                                                                     self.y_testing)
        validation_loss, validation_accuracy, validation_f1_score = self.evaluate_performance(cpuModel, self.x_validation,
                                                                                              self.y_validation)

        if epoch == 0:
            self.training_losses.append([])
            self.training_accuracies.append([])
            self.training_f1_scores.append([])

            self.testing_losses.append([])
            self.testing_accuracies.append([])
            self.testing_f1_scores.append([])

            self.validation_losses.append([])
            self.validation_accuracies.append([])
            self.validation_f1_scores.append([])

        self.training_losses[self.fold].append(training_loss.item())
        self.training_accuracies[self.fold].append(training_accuracy)
        self.training_f1_scores[self.fold].append(training_f1_score)

        self.testing_losses[self.fold].append(testing_loss.item())
        self.testing_accuracies[self.fold].append(testing_accuracy)
        self.testing_f1_scores[self.fold].append(testing_f1_score)

        self.validation_losses[self.fold].append(validation_loss.item())
        self.validation_accuracies[self.fold].append(validation_accuracy)
        self.validation_f1_scores[self.fold].append(validation_f1_score)

        if epoch % self.log_interval == 0:
            self.print_performance(self.fold, epoch)

    def evaluate_performance(self, model, inputs, correct_labels):
        model.eval()
        with torch.no_grad():

            outputs = model(inputs)
            if isinstance(self.loss_function, CustomCrossEntropyRegularisationTermLoss):
                loss = self.loss_function(outputs, correct_labels, self.model)
            else:
                loss = self.loss_function(outputs, correct_labels)

            _, predicted_labels = torch.max(outputs, 1)
            _, correct_labels = torch.max(correct_labels, 1)
            correct_predictions = (predicted_labels == correct_labels).sum().item()
            total_samples = len(correct_labels)
            accuracies = correct_predictions / total_samples * 100

            f1 = f1_score(correct_labels, predicted_labels, average='weighted')

        return loss, accuracies, f1

    def print_performance(self, fold, epoch):
        print("Date time: %s" % (datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
        print("Dataset: %s" % self.dataset_name)
        print("Method: %s" % self.method)
        print("Fold: %s" % fold)
        print("Epoch: %s" % epoch)
        print("+------------+--------------------+---------------------+------------------+")
        print("| Type       | Loss \t\t\t  | Accuracy \t\t\t| F1 scores: \t\t|")
        print("+------------+--------------------+---------------------+------------------+")
        print("| Training   | %s | %s \t| %s | " % (
            self.training_losses[-1][-1], self.training_accuracies[-1][-1], self.training_f1_scores[-1][-1]))
        print("+------------+--------------------+---------------------+-------------------+")
        print("| Testing    | %s \t| %s | %s | " % (
            self.testing_losses[-1][-1], self.testing_accuracies[-1][-1], self.testing_f1_scores[-1][-1]))
        print("+------------+--------------------+---------------------+-------------------+")
        print("| Validation | %s \t| %s | %s | " % (
            self.validation_losses[-1][-1], self.validation_accuracies[-1][-1], self.validation_f1_scores[-1][-1]))
        print("+------------+--------------------+---------------------+-------------------+")

    def log_performance(self, start_time, end_time, settings):
        run_object = {
            "method": self.method,
            "seed": self.seed,
            "runtime": format_runtime(start_time, end_time),
            "settings": settings.to_json_serializable(),
            "results": {
                "accuracies": {
                    "training": self.training_accuracies,
                    "testing": self.testing_accuracies,
                    "validation": self.validation_accuracies
                },
                "losses": {
                    "training": self.training_losses,
                    "testing": self.testing_losses,
                    "validation": self.validation_losses,
                },
                "f1_scores": {
                    "training": self.training_f1_scores,
                    "testing": self.testing_f1_scores,
                    "validation": self.validation_f1_scores,
                }
            }
        }
        return run_object

    def set_dataset(self, x_training, y_training, x_testing, y_testing, fold):
        self.x_training = x_training.to('cpu')
        self.y_training = y_training.to('cpu')
        self.x_testing = x_testing.to('cpu')
        self.y_testing = y_testing.to('cpu')
        self.fold = fold


def format_runtime(start_time, end_time):
    seconds = end_time - start_time
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{int(days)} days, {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds"
