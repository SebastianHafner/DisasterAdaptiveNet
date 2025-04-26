import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNCell, self).__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.in_gate = nn.Conv2d(input_size + hidden_size, hidden_size, 3, 1, 1)
        self.forget_gate = nn.Conv2d(input_size + hidden_size, hidden_size, 3, 1, 1)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, 3, 1, 1)
        self.cell_gate = nn.Conv2d(input_size + hidden_size, hidden_size, 3, 1, 1)

    def forward(self, input, h_state, c_state):
        conc_inputs = torch.cat((input, h_state), 1)

        in_gate = self.in_gate(conc_inputs)
        forget_gate = self.forget_gate(conc_inputs)
        out_gate = self.out_gate(conc_inputs)
        cell_gate = self.cell_gate(conc_inputs)

        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        out_gate = torch.sigmoid(out_gate)
        cell_gate = torch.tanh(cell_gate)

        c_state = (forget_gate * c_state) + (in_gate * cell_gate)
        h_state = out_gate * torch.tanh(c_state)

        return h_state, c_state


class DoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(InConv, self).__init__()
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(Down, self).__init__()

        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            conv_block(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(Up, self).__init__()

        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.detach().size()[2] - x1.detach().size()[2]
        diffX = x2.detach().size()[3] - x1.detach().size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SELayerPlus(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayerPlus, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.fc_combine = nn.Sequential(
            nn.Linear(channel * 2, channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        x1 = self.avg_pool(x).view(b, c)
        x1 = self.fc(x1)
        x3 = self.fc_combine(torch.cat((x1, x2), dim=1))
        x3 = x3.view(b, c, 1, 1)
        return x * x3.expand_as(x)


class ConditioningLayer(nn.Module):
    def __init__(self, channel_in: int, channel_feature: int = None):
        super(ConditioningLayer, self).__init__()
        channel_feature = channel_in if channel_feature is None else channel_feature
        self.fc = nn.Sequential(
            nn.Linear(channel_in, channel_feature, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel_feature, channel_feature, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        x3 = self.fc(x2)
        x3 = x3.view(b, c, 1, 1)
        return x * x3.expand_as(x)


class FiLM(nn.Module):
    # https://ojs.aaai.org/index.php/AAAI/article/view/11671
    def __init__(self, n_conditions: int, n_channels: int):
        super(FiLM, self).__init__()
        self.n_conditions = n_conditions
        self.n_channels = n_channels
        self.embeddings = nn.Embedding(n_conditions, n_channels)
        self.fc_gamma = nn.Sequential(
            nn.Linear(n_channels, n_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(n_channels, n_channels, bias=False),
        )
        self.fc_beta = nn.Sequential(
            nn.Linear(n_channels, n_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(n_channels, n_channels, bias=False),
        )

    def forward(self, feat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = feat.size()
        embed = torch.flatten(self.embeddings(x), start_dim=1)
        gamma = self.fc_gamma(embed).view(b, c, 1, 1)
        beta = self.fc_beta(embed).view(b, c, 1, 1)
        return feat * gamma.expand_as(feat) + beta.expand_as(feat)
