import torch
import torch.nn as nn
import torchvision.models as models
import torchvision

from collections import OrderedDict

class Vgg16Conv(nn.Module):
    """
    vgg16 convolution network architecture
    """

    def __init__(self, num_cls=1000):
        """
        Input
            number of class, default is 1k.
        """
        super(Vgg16Conv, self).__init__()
    
        self.features = nn.Sequential(
            # conv1 (input: 224x224x3)
            nn.Conv2d(3, 64, 3, padding=1),      # layer 0  -> out: 224x224x64
            nn.ReLU(),                           # layer 1
            nn.Conv2d(64, 64, 3, padding=1),     # layer 2  -> out: 224x224x64
            nn.ReLU(),                           # layer 3
            nn.MaxPool2d(2, stride=2, return_indices=True),  # layer 4  -> out: 112x112x64
            
            # conv2
            nn.Conv2d(64, 128, 3, padding=1),    # layer 5  -> out: 112x112x128
            nn.ReLU(),                           # layer 6
            nn.Conv2d(128, 128, 3, padding=1),   # layer 7  -> out: 112x112x128
            nn.ReLU(),                           # layer 8
            nn.MaxPool2d(2, stride=2, return_indices=True),  # layer 9  -> out: 56x56x128
            
            # conv3
            nn.Conv2d(128, 256, 3, padding=1),   # layer 10 -> out: 56x56x256
            nn.ReLU(),                           # layer 11
            nn.Conv2d(256, 256, 3, padding=1),   # layer 12 -> out: 56x56x256
            nn.ReLU(),                           # layer 13
            nn.Conv2d(256, 256, 3, padding=1),   # layer 14 -> out: 56x56x256
            nn.ReLU(),                           # layer 15
            nn.MaxPool2d(2, stride=2, return_indices=True),  # layer 16 -> out: 28x28x256

            # conv4
            nn.Conv2d(256, 512, 3, padding=1),   # layer 17 -> out: 28x28x512
            nn.ReLU(),                           # layer 18
            nn.Conv2d(512, 512, 3, padding=1),   # layer 19 -> out: 28x28x512
            nn.ReLU(),                           # layer 20
            nn.Conv2d(512, 512, 3, padding=1),   # layer 21 -> out: 28x28x512
            nn.ReLU(),                           # layer 22
            nn.MaxPool2d(2, stride=2, return_indices=True),  # layer 23 -> out: 14x14x512

            # conv5
            nn.Conv2d(512, 512, 3, padding=1),   # layer 24 -> out: 14x14x512
            nn.ReLU(),                           # layer 25
            nn.Conv2d(512, 512, 3, padding=1),   # layer 26 -> out: 14x14x512
            nn.ReLU(),                           # layer 27
            nn.Conv2d(512, 512, 3, padding=1),   # layer 28 -> out: 14x14x512
            nn.ReLU(),                           # layer 29
            nn.MaxPool2d(2, stride=2, return_indices=True)   # layer 30 -> out: 7x7x512
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_cls),
            nn.Softmax(dim=1)
        )

        # index of conv
        self.conv_layer_indices = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
        # feature maps
        self.feature_maps = OrderedDict()
        # switch
        self.pool_locs = OrderedDict()
        # initial weight
        self.init_weights()

    def init_weights(self):
        """
        initial weights from preptrained model by vgg16
        """
        vgg16_pretrained = models.vgg16(pretrained=True)
        # fine-tune Conv2d
        for idx, layer in enumerate(vgg16_pretrained.features):
            if isinstance(layer, nn.Conv2d):
                self.features[idx].weight.data = layer.weight.data
                self.features[idx].bias.data = layer.bias.data
        # fine-tune Linear
        for idx, layer in enumerate(vgg16_pretrained.classifier):
            if isinstance(layer, nn.Linear):
                self.classifier[idx].weight.data = layer.weight.data
                self.classifier[idx].bias.data = layer.bias.data
    
    def check(self):
        model = models.vgg16(pretrained=True)
        return model

    def forward(self, x):
        for idx, layer in enumerate(self.features):
            if isinstance(layer, nn.MaxPool2d):
                x, location = layer(x)
                # self.pool_locs[idx] = location
            else:
                x = layer(x)
        
        # reshape to (1, 512 * 7 * 7)
        x = x.view(x.size()[0], -1)
        output = self.classifier(x)
        return output

if __name__ == '__main__':
    model = models.vgg16(pretrained=True)
    print(model)
