import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, dim_z, num_classes):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(dim_z, num_classes)

    def forward(self, x):
        x = F.relu(self.linear(x))
        return x



