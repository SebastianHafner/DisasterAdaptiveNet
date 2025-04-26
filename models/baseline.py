import abc
from typing import Sequence
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models
from torchvision.models import ResNet34_Weights
import timm

from utils.experiment_manager import CfgNode
from models import building_blocks


class ConvRelu(nn.Module):
    """ Convolution -> ReLU.

        Args:
            in_channels : number of input channels
            out_channels : number of output channels
            kernel_size : size of convolution kernel
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super(ConvRelu, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class LocModel(nn.Module, metaclass=abc.ABCMeta):
    """ Base class for all localization models."""

    def __init__(self, cfg: CfgNode) -> None:
        super().__init__()
        self.cfg = cfg

    @abc.abstractmethod
    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.forward_once(x))

    def _initialize_weights(self) -> None:
        """ Initialize weights of the model."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Res34_Unet_Loc(LocModel):
    """Unet model with a resnet34 encoder used for localization.
        Args:
            pretrained : if True, use pretrained resnet34 weights.

    """

    def __init__(self, cfg: CfgNode) -> None:
        super(Res34_Unet_Loc, self).__init__(cfg)
        pretrained: bool = True

        encoder_filters = [64, 64, 128, 256, 512]
        decoder_filters = np.asarray([48, 64, 96, 160, 320])

        self.conv6 = ConvRelu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = ConvRelu(decoder_filters[-1] + encoder_filters[-2], decoder_filters[-1])
        self.conv7 = ConvRelu(decoder_filters[-1], decoder_filters[-2])
        self.conv7_2 = ConvRelu(decoder_filters[-2] + encoder_filters[-3], decoder_filters[-2])
        self.conv8 = ConvRelu(decoder_filters[-2], decoder_filters[-3])
        self.conv8_2 = ConvRelu(decoder_filters[-3] + encoder_filters[-4], decoder_filters[-3])
        self.conv9 = ConvRelu(decoder_filters[-3], decoder_filters[-4])
        self.conv9_2 = ConvRelu(decoder_filters[-4] + encoder_filters[-5], decoder_filters[-4])
        self.conv10 = ConvRelu(decoder_filters[-4], decoder_filters[-5])

        self.res = nn.Conv2d(decoder_filters[-5], 1, 1, stride=1, padding=0)

        self._initialize_weights()
        # pretrained argument was deprecated so we changed to weights

        # throw error if pretrained is not a bool
        if not isinstance(pretrained, bool):
            raise TypeError("pretrained argument should be a bool")
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        if weights is not None:
            print(f"using weights from {weights}")
        encoder = torchvision.models.resnet34(weights=weights)
        self.conv1 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.conv2 = nn.Sequential(encoder.maxpool, encoder.layer1)
        self.conv3 = encoder.layer2
        self.conv4 = encoder.layer3
        self.conv5 = encoder.layer4

    def encoder_once(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)
        return enc1, enc2, enc3, enc4, enc5

    def decoder_once(self, x_enc: Sequence[torch.Tensor]) -> torch.Tensor:
        enc1, enc2, enc3, enc4, enc5 = x_enc
        dec6 = self.conv6(F.interpolate(enc5, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc4], 1))
        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc3], 1))
        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc2], 1))
        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))
        dec9 = self.conv9_2(torch.cat([dec9, enc1], 1))
        dec10 = self.conv10(F.interpolate(dec9, scale_factor=2))
        return dec10

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        x_enc = self.encoder_once(x)
        x_dec = self.decoder_once(x_enc)
        return x_dec


class Siamese(nn.Module, metaclass=abc.ABCMeta, ):
    """ Abstract class for siamese networks. To create a
    siamese network, inherit from this class and the class of the localization model.
    """

    def __init__(self, cfg) -> None:
        super(Siamese, self).__init__(cfg)
        decoder_filters = np.asarray([48, 64, 96, 160, 320])
        self.res = nn.Conv2d(decoder_filters[-5] * 2, self.cfg.MODEL.OUT_CHANNELS, 1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass  for two inputs (that have been concateneted chanelwise to one). """
        output1 = self.forward_once(x[:, :3, :, :])
        output2 = self.forward_once(x[:, 3:, :, :])
        return self.res(torch.cat([output1, output2], 1))


class Res34_Unet_Double(Siamese, Res34_Unet_Loc):
    """ ResNet34 Unet model for classification tasks."""

    def __init__(self, cfg):
        super(Res34_Unet_Double, self).__init__(cfg)

    def encode_once(self, x: torch.Tensor) -> torch.Tensor:
        """ Encode one image with the encoder part of the model."""
        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)
        return enc5

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """ Get the embeddings of the images."""
        encoded1 = self.encode_once(x[:, :3, :, :])
        encoded2 = self.encode_once(x[:, 3:, :, :])
        encoded = torch.cat([encoded1, encoded2], 1)
        return F.adaptive_avg_pool2d(encoded, 1).view(encoded.shape[0], -1)


class DisasterTypeNet(Siamese, Res34_Unet_Loc):
    """ ResNet34 Unet model for classification tasks."""

    def __init__(self, cfg) -> None:
        super(DisasterTypeNet, self).__init__(cfg)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass  for two inputs (that have been concateneted chanelwise to one). """
        output1 = self.forward_once(x[:, :3])
        output2 = self.forward_once(x[:, 3:])
        output = self.res(torch.cat([output1, output2], 1))
        output = torch.flatten(self.avg_pool(output), start_dim=1)
        return output


class StrongBaselineNet(Siamese, Res34_Unet_Loc):
    """ ResNet34 Unet model for classification tasks."""

    def __init__(self, cfg) -> None:
        super(StrongBaselineNet, self).__init__(cfg)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass  for two inputs (that have been concateneted chanelwise to one). """
        output1 = self.forward_once(x[:, :3, :, :])
        output2 = self.forward_once(x[:, 3:, :, :])
        return self.res(torch.cat([output1, output2], 1))


class DisasterAdaptiveNet(Siamese, Res34_Unet_Loc):
    """ ResNet34 Unet model for classification tasks."""

    def __init__(self, cfg) -> None:
        super(DisasterAdaptiveNet, self).__init__(cfg)
        decoder_filters = np.asarray([48, 64, 96, 160, 320])
        self.conditioning_layer = building_blocks.FiLM(len(cfg.DATASET.CONDITIONING_KEY), decoder_filters[0])

    def forward(self, x: torch.Tensor, lookup_tensor: torch.Tensor = None) -> torch.Tensor:
        """ Forward pass  for two inputs (that have been concateneted chanelwise to one). """
        output1 = self.forward_once(x[:, :3, :, :])
        output2 = self.forward_once(x[:, 3:, :, :])
        if lookup_tensor is not None:
            output2 = self.conditioning_layer(output2, lookup_tensor)
        return self.res(torch.cat([output1, output2], 1))


class DisasterSpecificNets(nn.Module):

    def __init__(self, cfg) -> None:
        super(DisasterSpecificNets, self).__init__()
        self.cfg = cfg
        self.n_disaster_types = len(cfg.DATASET.CONDITIONING_KEY)
        self.models = nn.ModuleList([StrongBaselineNet(cfg) for _ in range(self.n_disaster_types)])

    def forward(self, x: torch.Tensor, lookup_table: torch.Tensor = None) -> torch.Tensor:
        B, _, H, W = x.size()
        output = torch.empty(((B, self.cfg.MODEL.OUT_CHANNELS, H, W)), device=x.device)
        for i in range(self.n_disaster_types):
            indices = (lookup_table == i).nonzero(as_tuple=True)[0]
            if indices.numel() > 0:
                x_disaster = x[indices]
                output_disaster = self.models[i].forward(x_disaster)
                output[indices] = output_disaster
        return output


class Swin_Unet_Loc(nn.Module):
    """ Unet model with a Swin Transformer encoder used for localization."""

    def __init__(self, cfg: CfgNode) -> None:
        super(Swin_Unet_Loc, self).__init__()
        self.cfg = cfg
        self.encoder = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, features_only=True)

        encoder_filters = [96, 192, 384, 768]  # Swin-T feature map sizes
        decoder_filters = [48, 64, 96, 160, 320]

        self.conv6 = nn.Conv2d(encoder_filters[-1], decoder_filters[-1], kernel_size=3, padding=1)
        self.conv6_2 = nn.Conv2d(decoder_filters[-1] + encoder_filters[-2], decoder_filters[-1], kernel_size=3,
                                 padding=1)
        self.conv7 = nn.Conv2d(decoder_filters[-1], decoder_filters[-2], kernel_size=3, padding=1)
        self.conv7_2 = nn.Conv2d(decoder_filters[-2] + encoder_filters[-3], decoder_filters[-2], kernel_size=3,
                                 padding=1)
        self.conv8 = nn.Conv2d(decoder_filters[-2], decoder_filters[-3], kernel_size=3, padding=1)
        self.conv8_2 = nn.Conv2d(decoder_filters[-3] + encoder_filters[-4], decoder_filters[-3], kernel_size=3,
                                 padding=1)

        self.res = nn.Conv2d(decoder_filters[-3], 1, kernel_size=1)

    def encoder_once(self, x: torch.Tensor):
        features = self.encoder(x)
        return features  # List of feature maps at different levels

    def decoder_once(self, x_enc):
        enc1, enc2, enc3, enc4 = x_enc  # Swin outputs 4 feature maps

        dec6 = self.conv6(F.interpolate(enc4, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc3], dim=1))
        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc2], dim=1))
        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc1], dim=1))

        return self.res(F.interpolate(dec8, scale_factor=2))

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        x_enc = self.encoder_once(x)
        x_dec = self.decoder_once(x_enc)
        return x_dec


class StrongBaselineNetSwin(nn.Module):
    """ Strong baseline network using Swin Transformer as encoder."""

    def __init__(self, cfg: CfgNode) -> None:
        super(StrongBaselineNetSwin, self).__init__()
        self.unet = Swin_Unet_Loc(cfg)
        self.res = nn.Conv2d(320 * 2, cfg.MODEL.OUT_CHANNELS, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output1 = self.unet.forward_once(x[:, :3, :, :])
        output2 = self.unet.forward_once(x[:, 3:, :, :])
        return self.res(torch.cat([output1, output2], dim=1))


class ConvNeXt_Unet_Loc(nn.Module):
    """ Unet model with a ConvNeXt encoder used for localization."""

    def __init__(self, cfg: CfgNode) -> None:
        super(ConvNeXt_Unet_Loc, self).__init__()
        self.cfg = cfg
        self.encoder = timm.create_model("convnext_tiny", pretrained=True, features_only=True)

        encoder_filters = [96, 192, 384, 768]  # ConvNeXt-T feature map sizes
        decoder_filters = [48, 64, 96, 160, 320]

        self.conv6 = nn.Conv2d(encoder_filters[-1], decoder_filters[-1], kernel_size=3, padding=1)
        self.conv6_2 = nn.Conv2d(decoder_filters[-1] + encoder_filters[-2], decoder_filters[-1], kernel_size=3,
                                 padding=1)
        self.conv7 = nn.Conv2d(decoder_filters[-1], decoder_filters[-2], kernel_size=3, padding=1)
        self.conv7_2 = nn.Conv2d(decoder_filters[-2] + encoder_filters[-3], decoder_filters[-2], kernel_size=3,
                                 padding=1)
        self.conv8 = nn.Conv2d(decoder_filters[-2], decoder_filters[-3], kernel_size=3, padding=1)
        self.conv8_2 = nn.Conv2d(decoder_filters[-3] + encoder_filters[-4], decoder_filters[-3], kernel_size=3,
                                 padding=1)

        self.res = nn.Conv2d(decoder_filters[-3], 1, kernel_size=1)

    def encoder_once(self, x: torch.Tensor):
        features = self.encoder(x)
        return features  # List of feature maps at different levels

    def decoder_once(self, x_enc):
        enc1, enc2, enc3, enc4 = x_enc  # ConvNeXt outputs 4 feature maps

        dec6 = self.conv6(F.interpolate(enc4, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc3], dim=1))
        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc2], dim=1))
        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc1], dim=1))

        return self.res(F.interpolate(dec8, scale_factor=2))

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        x_enc = self.encoder_once(x)
        x_dec = self.decoder_once(x_enc)
        return x_dec


class StrongBaselineNetConvNeXt(nn.Module):
    """ Strong baseline network using ConvNeXt as encoder."""

    def __init__(self, cfg: CfgNode) -> None:
        super(StrongBaselineNetConvNeXt, self).__init__()
        self.unet = ConvNeXt_Unet_Loc(cfg)
        self.res = nn.Conv2d(320 * 2, cfg.MODEL.OUT_CHANNELS, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output1 = self.unet.forward_once(x[:, :3, :, :])
        output2 = self.unet.forward_once(x[:, 3:, :, :])
        return self.res(torch.cat([output1, output2], dim=1))
