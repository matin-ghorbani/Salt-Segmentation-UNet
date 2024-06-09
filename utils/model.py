import torch
from torch import nn
from torch.nn import functional as f
from torchvision import transforms

from . import config


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return self.conv2(x)


class Encoder(nn.Module):
    def __init__(self, channels: tuple[int, int, int, int] = (3, 16, 32, 64), *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.encoder_blocks = nn.ModuleList([
            Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)
        ])
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        # Store the intermediate outputs
        block_outputs = []

        for block in self.encoder_blocks:
            x = block(x)
            block_outputs.append(x)
            x = self.max_pool(x)

        return block_outputs


class Decoder(nn.Module):
    def __init__(self, channels: tuple[int, int, int] = (64, 32, 16), *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.channels = channels
        self.up_convs = nn.ModuleList([
            nn.ConvTranspose2d(channels[i], channels[i + 1], 2, 2) for i in range(len(channels) - 1)
        ])

        self.decoder_blocks = nn.ModuleList([
            Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)
        ])

    def crop(self, x, encoder_features):
        h, w = x.shape[2:]
        return transforms.CenterCrop((h, w))(encoder_features)

    def forward(self, x, encoder_features):
        for i in range(len(self.channels) - 1):
            x = self.up_convs[i](x)

            encoder_feature = self.crop(x, encoder_features[i])
            x = torch.cat((x, encoder_feature), 1)
            x = self.decoder_blocks[i](x)

        return x


class UNet(nn.Module):
    def __init__(
        self,
        encoder_channels: tuple[int, int, int, int] = (3, 16, 32, 64),
        decoder_channels: tuple[int, int, int] = (64, 32, 16),
        classes_num: int = 1,
        retain_dim: bool = True,
        out_size: tuple[int, int] = (
            config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH),
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.encoder = Encoder(encoder_channels)
        self.decoder = Decoder(decoder_channels)

        self.head = nn.Conv2d(decoder_channels[-1], classes_num, 1)
        self.retain_dim = retain_dim
        self.out_size = out_size

    def forward(self, x):
        encoder_features = self.encoder(x)

        # Pass the encoder features through decoder making sure that
        # Their dimensions are suited for concatenation

        # encoder_features list contains all the feature maps starting from the first encoder block output to the last
        
        # We reverse the order of feature maps in this list
        reversed_encoder_features = encoder_features[::-1]
        decoder_features = self.decoder(

            # Pass the output of the final encoder block
            reversed_encoder_features[0],
            # Feature map outputs of all intermediate encoder blocks 
            reversed_encoder_features[1:]
        )

        # Pass the decoder output to convolution head to obtain the segmentation mask
        segmentation_map = self.head(decoder_features)

        if self.retain_dim:
            segmentation_map = f.interpolate(segmentation_map, self.out_size)

        return segmentation_map
