import torch
from classifiers import distance_calculator


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        return distance_calculator.calculate_distance(output1, output2, 'euclidean')
