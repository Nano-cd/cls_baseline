from torchvision.models import squeezenet1_0
import torch.nn as nn
import torch


class Shuffle_net(nn.Module):
    def __init__(self, num_classes):
        super(Shuffle_net, self).__init__()
        base_model = squeezenet1_0(pretrained=True)
        self.base = nn.Sequential(*list(base_model.children())[:-2])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(192, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        features = self.base(x)
        out = self.avg_pool(features)
        out = torch.flatten(out, 1)
        activation = out
        out = self.fc(out)
        return out, activation
