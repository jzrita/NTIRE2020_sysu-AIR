import torch
import torch.nn as nn
from .blocks import ConvBlock, DeconvBlock, MeanShift


def mean_channels(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


# contrast-aware channel attention module
class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class IMDModule(nn.Module):
    def __init__(self, in_channels, act_type, norm_type, distillation_rate=0.25):
        super(IMDModule, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        self.c1 = ConvBlock(in_channels, in_channels, kernel_size=1, act_type=act_type, norm_type=None)
        self.c2 = ConvBlock(self.remaining_channels, in_channels, kernel_size=3, act_type=act_type, norm_type=None)
        self.c3 = ConvBlock(self.remaining_channels, in_channels, kernel_size=3, act_type=act_type, norm_type=None)
        self.c4 = ConvBlock(self.remaining_channels, self.distilled_channels, kernel_size=3, act_type=None,
                            norm_type=norm_type)
        self.c5 = ConvBlock(in_channels, in_channels, kernel_size=1, act_type=None, norm_type=None)
        self.cca = CCALayer(self.distilled_channels * 4)

    def forward(self, input):
        out_c1 = self.c1(input)
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c2 = self.c2(remaining_c1)
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c3 = self.c3(remaining_c2)
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c4 = self.c4(remaining_c3)
        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
        out_fused = self.c5(self.cca(out)) + input
        return out_fused


class IMDBlock(nn.Module):
    def __init__(self, num_features, num_groups, act_type, norm_type):
        super(IMDBlock, self).__init__()
        self.compress_in = ConvBlock(2 * num_features, num_features, kernel_size=1, act_type=act_type,
                                     norm_type=norm_type)
        self.IMDMs = []
        for _ in range(num_groups):
            self.IMDMs.append(IMDModule(in_channels=num_features, act_type=act_type, norm_type=norm_type))
        self.IMDMs = nn.Sequential(*self.IMDMs)

        self.should_reset = True
        self.last_hidden = None

    def forward(self, input):
        if self.should_reset:
            self.last_hidden = torch.zeros(input.size()).cuda()
            # self.last_hidden = torch.zeros(input.size())
            self.last_hidden.copy_(input)
            self.should_reset = False

        input = torch.cat((input, self.last_hidden), dim=1)
        input = self.compress_in(input)
        output = self.IMDMs(input)
        self.last_hidden = output
        return output

    def reset_state(self):
        self.should_reset = True


class CAON(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, num_steps, num_groups, upscale_factor, act_type='prelu',
                 norm_type=None):
        super(CAON, self).__init__()

        if upscale_factor == 2:
            stride = 2
            padding = 2
            kernel_size = 6
        elif upscale_factor == 3:
            stride = 3
            padding = 2
            kernel_size = 7
        elif upscale_factor == 4:
            stride = 4
            padding = 2
            kernel_size = 8
        elif upscale_factor == 8:
            stride = 8
            padding = 2
            kernel_size = 12
        elif upscale_factor == 16:
            stride = 16
            padding = 2
            kernel_size = 20

        self.num_steps = num_steps
        self.num_features = num_features
        self.upscale_factor = upscale_factor

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_mean, rgb_std)

        # LR feature extraction block
        self.conv_in = ConvBlock(in_channels, 4 * num_features,
                                 kernel_size=3,
                                 act_type=act_type, norm_type=norm_type)
        self.feat_in = ConvBlock(4 * num_features, num_features,
                                 kernel_size=1,
                                 act_type=act_type, norm_type=norm_type)

        # basic block
        self.block = IMDBlock(num_features, num_groups, act_type, norm_type)

        # reconstruction block
        # uncomment for pytorch 0.4.0
        # self.upsample = nn.Upsample(scale_factor=upscale_factor, mode='bilinear')

        self.out = DeconvBlock(num_features, num_features,
                               kernel_size=kernel_size, stride=stride, padding=padding,
                               act_type='prelu', norm_type=norm_type)
        self.conv_out = ConvBlock(num_features, out_channels,
                                  kernel_size=3,
                                  act_type=None, norm_type=norm_type)

        self.add_mean = MeanShift(rgb_mean, rgb_std, 1)

    def forward(self, x):
        self._reset_state()

        x = self.sub_mean(x)
        # uncomment for pytorch 0.4.0
        # inter_res = self.upsample(x)

        # comment for pytorch 0.4.0
        inter_res = nn.functional.interpolate(x, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)

        x = self.conv_in(x)
        x = self.feat_in(x)

        outs = []
        for _ in range(self.num_steps):
            h = self.block(x)

            h = torch.add(inter_res, self.conv_out(self.out(h)))
            h = self.add_mean(h)
            outs.append(h)

        return outs  # return output of every timesteps

    def _reset_state(self):
        self.block.reset_state()
