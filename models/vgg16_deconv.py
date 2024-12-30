import torch
import torch.nn as nn
import torchvision.models as models

import sys

class Vgg16Deconv(nn.Module):
    """
    vgg16 transpose convolution network architecture
    """
    def __init__(self):
        super(Vgg16Deconv, self).__init__()

        self.features = nn.Sequential(
            # deconv1 (starting from 7x7x512)
            nn.MaxUnpool2d(2, stride=2),                 # 0  | 7x7x512 -> 14x14x512
            nn.ReLU(),                                   # 1  | shape unchanged
            nn.ConvTranspose2d(512, 512, 3, padding=1),  # 2  | shape unchanged
            nn.ReLU(),                                   # 3  | shape unchanged
            nn.ConvTranspose2d(512, 512, 3, padding=1),  # 4  | shape unchanged
            nn.ReLU(),                                   # 5  | shape unchanged
            nn.ConvTranspose2d(512, 512, 3, padding=1),  # 6  | shape unchanged

            # deconv2 
            nn.MaxUnpool2d(2, stride=2),                 # 7  | 14x14x512 -> 28x28x512
            nn.ReLU(),                                   # 8  | shape unchanged
            nn.ConvTranspose2d(512, 512, 3, padding=1),  # 9  | shape unchanged
            nn.ReLU(),                                   # 10 | shape unchanged
            nn.ConvTranspose2d(512, 512, 3, padding=1),  # 11 | shape unchanged
            nn.ReLU(),                                   # 12 | shape unchanged
            nn.ConvTranspose2d(512, 256, 3, padding=1),  # 13 | 28x28x512 -> 28x28x256
            
            # deconv3
            nn.MaxUnpool2d(2, stride=2),                 # 14 | 28x28x256 -> 56x56x256
            nn.ReLU(),                                   # 15 | shape unchanged
            nn.ConvTranspose2d(256, 256, 3, padding=1),  # 16 | shape unchanged
            nn.ReLU(),                                   # 17 | shape unchanged
            nn.ConvTranspose2d(256, 256, 3, padding=1),  # 18 | shape unchanged
            nn.ReLU(),                                   # 19 | shape unchanged
            nn.ConvTranspose2d(256, 128, 3, padding=1),  # 20 | 56x56x256 -> 56x56x128
            
            # deconv4
            nn.MaxUnpool2d(2, stride=2),                 # 21 | 56x56x128 -> 112x112x128
            nn.ReLU(),                                   # 22 | shape unchanged
            nn.ConvTranspose2d(128, 128, 3, padding=1),  # 23 | shape unchanged
            nn.ReLU(),                                   # 24 | shape unchanged
            nn.ConvTranspose2d(128, 64, 3, padding=1),   # 25 | 112x112x128 -> 112x112x64
            
            # deconv5
            nn.MaxUnpool2d(2, stride=2),                 # 26 | 112x112x64 -> 224x224x64
            nn.ReLU(),                                   # 27 | shape unchanged
            nn.ConvTranspose2d(64, 64, 3, padding=1),    # 28 | shape unchanged
            nn.ReLU(),                                   # 29 | shape unchanged
            nn.ConvTranspose2d(64, 3, 3, padding=1)      # 30 | 224x224x64 -> 224x224x3
        )

        self.conv2deconv_indices = {
            0:30,   # First conv  (224x224) → Last deconv   (3x3)
            2:28,   # Second conv (224x224) → Second-to-last deconv
            5:25,   # Third conv  (112x112) → Third deconv  (128→64)
            7:23,   # Fourth conv (112x112) → Fourth deconv (128→128)
            10:20,  # Fifth conv  (56x56)   → Fifth deconv  (256→128)
            12:18,  # Sixth conv  (56x56)   → Sixth deconv  (256→256)
            14:16,  # Seventh conv(28x28)   → Seventh deconv(256→256)
            17:13,  # Eighth conv (28x28)   → Eighth deconv (512→256)
            19:11,  # Ninth conv  (14x14)   → Ninth deconv  (512→512)
            21:9,   # Tenth conv  (14x14)   → Tenth deconv  (512→512)
            24:6,   # 11th conv   (7x7)     → 11th deconv   (512→512)
            26:4,   # 12th conv   (7x7)     → 12th deconv   (512→512)
            28:2    # 13th conv   (7x7)     → 13th deconv   (512→512)
        }

        self.unpool2pool_indices = {
            26:4,   # Last unpool    → First pool  (224→112)
            21:9,   # Fourth unpool  → Second pool (112→56)
            14:16,  # Third unpool   → Third pool  (56→28)
            7:23,   # Second unpool  → Fourth pool (28→14)
            0:30    # First unpool   → Fifth pool  (14→7)
        }

        self.init_weight()

    def init_weight(self):
        vgg16_pretrained = models.vgg16(pretrained=True)
        for idx, layer in enumerate(vgg16_pretrained.features):
            if isinstance(layer, nn.Conv2d):
                self.features[self.conv2deconv_indices[idx]].weight.data = layer.weight.data
                #self.features[self.conv2deconv_indices[idx]].bias.data\
                # = layer.bias.data
        
    def forward(self, x, layer, activation_idx, pool_locs):
        # 1. Find where to start deconvolution
        if layer in self.conv2deconv_indices:
            # Get corresponding deconv layer index
            start_idx = self.conv2deconv_indices[layer]
            # Example: if layer=0, start_idx=30 (start from last deconv layer)
        else:
            raise ValueError('layer is not a conv feature map')

        # 2. Process through deconv layers
        for idx in range(start_idx, len(self.features)):
            # If we hit an unpooling layer
            if isinstance(self.features[idx], nn.MaxUnpool2d):
                # Example: if idx=26 (last unpool)
                # Get pool indices from pool_locs[4] (first pool)
                x = self.features[idx]\
                (x, pool_locs[self.unpool2pool_indices[idx]])
            else:
                # For ConvTranspose2d and ReLU, just apply normally
                x = self.features[idx](x)
        return x
