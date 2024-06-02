import math

import torch
import torch.nn as nn


class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, predictions, targets):
        # Calculate cross-entropy loss
        output = self.loss(predictions, targets)
        if math.isnan(output):
            print(predictions)
            print(targets)
            raise Exception("Not a number")
        if math.isinf(output):
            print(predictions)
            print(targets)
            raise Exception("Inf number")
        return output


class CustomMSELoss(nn.Module):
    def __init__(self):
        super(CustomMSELoss, self).__init__()

    def forward(self, predictions, targets):
        # Apply softmax to convert logits into probabilities
        predictions = torch.softmax(predictions, dim=1)
        # Calculate mse loss
        predictions = torch.argmax(predictions, axis=1)
        targets = torch.argmax(targets, axis=1)
        difference = torch.subtract(predictions, targets)
        loss = torch.sum(torch.multiply(difference, difference))
        return loss


class CustomCrossEntropyRegularisationTermLoss(nn.Module):
    def __init__(self, lambda_):
        super(CustomCrossEntropyRegularisationTermLoss, self).__init__()
        self.lambda_ = lambda_
        self.loss = nn.CrossEntropyLoss()

    def forward(self, predictions, targets, model):
        l2RegularizationTerm = sum(torch.sum(param ** 2) for param in model.parameters())
        output = self.loss(predictions, targets) + 0.5 * self.lambda_ * l2RegularizationTerm
        if math.isnan(output):
            raise Exception("Not a number")
        if math.isinf(output):
            raise Exception("Inf number")
        return output
