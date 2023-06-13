import torch
import torch.nn as nn
import torch.nn.functional as F

BN_MOMENTUM = 0.1
ALIGN_CORNERS = None


class PAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class DA_Module(nn.Module):
    def __init__(self, in_channel):
        super(DA_Module, self).__init__()
        self.pam = PAM_Module(in_channel)  # 位置注意力模块
        self.cam = CAM_Module(in_channel)  # 通道注意力模块

    def forward(self, x):
        sa_pam = self.pam(x)
        sa_cam = self.cam(x)
        fusion = sa_cam + sa_pam
        return fusion


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

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class Smooth_Fusion(nn.Module):
    def __init__(self, cexpand, cfusion):
        super(Smooth_Fusion, self).__init__()

        self.expand = nn.Sequential(
            nn.Conv2d(cexpand, cfusion, kernel_size=3, padding=1),
            nn.BatchNorm2d(cfusion),
            nn.ReLU(inplace=True)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(cfusion, cexpand, kernel_size=3, padding=1),
            nn.BatchNorm2d(cexpand),
            nn.ReLU(inplace=True),
            nn.Conv2d(cexpand, cexpand, kernel_size=3, padding=1),
            nn.BatchNorm2d(cexpand),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, y):
        if x.shape[1] == y.shape[1]:
            out = self.expand(x) + y
        else:
            out = self.expand(x) + F.interpolate(y, scale_factor=2, mode='bilinear',
                                                 align_corners=ALIGN_CORNERS)
        out = self.fusion(out)
        return out


class CFRM(nn.Module):
    def __init__(self, in_channel):
        super(CFRM, self).__init__()
        self.se_rgb = SELayer(in_channel)
        self.se_msi = SELayer(in_channel)

    def forward(self, x, y):
        recalibration_rgb = self.se_rgb(x)
        recalibration_msi = self.se_rgb(y)
        fused = recalibration_rgb + recalibration_msi
        return fused


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class StageModule(nn.Module):
    def __init__(self, input_branches, output_branches, c):
        """
        构建对应stage，即用来融合不同尺度的实现
        :param input_branches: 输入的分支数，每个分支对应一种尺度
        :param output_branches: 输出的分支数
        :param c: 输入的第一个分支通道数
        """
        super().__init__()
        self.input_branches = input_branches
        self.output_branches = output_branches

        self.branches = nn.ModuleList()
        for i in range(self.input_branches):  # 每个分支上都先通过4个BasicBlock
            w = c * (2 ** i)  # 对应第i个分支的通道数
            branch = nn.Sequential(
                BasicBlock(w, w),
                BasicBlock(w, w),
                BasicBlock(w, w),
                BasicBlock(w, w)
            )
            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()  # 用于融合每个分支上的输出
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.input_branches):
                if i == j:
                    # 当输入、输出为同一个分支时不做任何处理
                    self.fuse_layers[-1].append(nn.Identity())
                elif i < j:
                    # 当输入分支j大于输出分支i时(即输入分支下采样率大于输出分支下采样率)，
                    # 此时需要对输入分支j进行通道调整以及上采样，方便后续相加
                    self.fuse_layers[-1].append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=1, stride=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM),
                            nn.Upsample(scale_factor=2.0 ** (j - i), mode='nearest')
                        )
                    )
                else:  # i > j
                    # 当输入分支j小于输出分支i时(即输入分支下采样率小于输出分支下采样率)，
                    # 此时需要对输入分支j进行通道调整以及下采样，方便后续相加
                    # 注意，这里每次下采样2x都是通过一个3x3卷积层实现的，4x就是两个，8x就是三个，总共i-j个
                    ops = []
                    # 前i-j-1个卷积层不用变通道，只进行下采样
                    for k in range(i - j - 1):
                        ops.append(
                            nn.Sequential(
                                nn.Conv2d(c * (2 ** j), c * (2 ** j), kernel_size=3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(c * (2 ** j), momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)
                            )
                        )
                    # 最后一个卷积层不仅要调整通道，还要进行下采样
                    ops.append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM)
                        )
                    )
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 每个分支通过对应的block
        x = [branch(xi) for branch, xi in zip(self.branches, x)]

        # 接着融合不同尺寸信息
        x_fused = []
        for i in range(len(self.fuse_layers)):
            x_fused.append(
                self.relu(
                    sum([self.fuse_layers[i][j](x[j]) for j in range(len(self.branches))])
                )
            )

        return x_fused


class HighResolutionNet(nn.Module):
    def __init__(self, base_channel: int = 32, num_classes: int = 2):
        super().__init__()
        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        # Stage1
        downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM)
        )
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=downsample),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64)
        )

        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, base_channel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(base_channel, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Sequential(  # 这里又使用一次Sequential是为了适配原项目中提供的权重
                    nn.Conv2d(256, base_channel * 2, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(base_channel * 2, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                )
            )
        ])

        # Stage2
        self.stage2 = nn.Sequential(
            StageModule(input_branches=2, output_branches=2, c=base_channel)
        )

        # transition2
        self.transition2 = nn.ModuleList([
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(base_channel * 2, base_channel * 4, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(base_channel * 4, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                )
            )
        ])

        # Stage3
        self.stage3 = nn.Sequential(
            StageModule(input_branches=3, output_branches=3, c=base_channel),
            StageModule(input_branches=3, output_branches=3, c=base_channel),
            StageModule(input_branches=3, output_branches=3, c=base_channel),
            StageModule(input_branches=3, output_branches=3, c=base_channel)
        )

        # transition3
        self.transition3 = nn.ModuleList([
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(base_channel * 4, base_channel * 8, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(base_channel * 8, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                )
            )
        ])

        # Stage4
        self.stage4 = nn.Sequential(
            StageModule(input_branches=4, output_branches=4, c=base_channel),
            StageModule(input_branches=4, output_branches=4, c=base_channel),
            StageModule(input_branches=4, output_branches=4, c=base_channel)
        )

        self.final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=base_channel * 5,
                out_channels=base_channel * 2,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(base_channel * 2, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=base_channel * 2,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0)
        )

        # msi part
        self.msiconv1 = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.Conv2d(64, base_channel * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channel * 2, momentum=BN_MOMENTUM)
        )
        self.msiconv2 = nn.Sequential(
            nn.Conv2d(base_channel * 2, base_channel * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channel * 4, momentum=BN_MOMENTUM),
        )
        self.msiconv3 = nn.Sequential(
            nn.Conv2d(base_channel * 4, base_channel * 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channel * 8, momentum=BN_MOMENTUM),
        )

        # 生成金字塔特征
        self.pyramid_1 = nn.Sequential(
            nn.Conv2d(64, base_channel * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channel * 2, momentum=BN_MOMENTUM),
        )
        self.pyramid_2 = nn.Sequential(
            nn.Conv2d(base_channel * 2, base_channel * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channel * 4, momentum=BN_MOMENTUM),
        )
        self.pyramid_3 = nn.Sequential(
            nn.Conv2d(base_channel * 4, base_channel * 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channel * 8, momentum=BN_MOMENTUM),
        )

        # Cross-source Feature Recalibration Module
        self.CFRM1 = CFRM(base_channel * 2)
        self.CFRM2 = CFRM(base_channel * 4)
        self.CFRM3 = CFRM(base_channel * 8)

        # Dual att
        self.DA32 = DA_Module(base_channel * 8)

        self.SFM1 = Smooth_Fusion(base_channel * 8, base_channel * 8)
        self.SFM2 = Smooth_Fusion(base_channel * 4, base_channel * 8)
        self.SFM3 = Smooth_Fusion(base_channel * 2, base_channel * 4)
        self.SFM4 = Smooth_Fusion(base_channel * 1, base_channel * 2)

        self.toplayer = nn.Conv2d(base_channel * 8, base_channel, kernel_size=1, stride=1, padding=0)  # Reduce channels
        self.latlayer1 = nn.Conv2d(base_channel * 4, base_channel, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(base_channel * 2, base_channel, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(base_channel, base_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x, y):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # RGB
        rgb_pyramid_128 = self.pyramid_1(x)
        rgb_pyramid_64 = self.pyramid_2(rgb_pyramid_128)
        rgb_pyramid_32 = self.pyramid_3(rgb_pyramid_64)

        # MSI
        msi_pyramid_128 = self.msiconv1(y)
        msi_pyramid_64 = self.pyramid_2(msi_pyramid_128)
        msi_pyramid_32 = self.pyramid_3(msi_pyramid_64)

        # Recalibration
        Recalibration_128 = self.CFRM1(rgb_pyramid_128, msi_pyramid_128)
        Recalibration_64 = self.CFRM2(rgb_pyramid_64, msi_pyramid_64)
        Recalibration_32 = self.CFRM3(rgb_pyramid_32, msi_pyramid_32)

        x = self.layer1(x)
        x = [x, x]
        x = [
            self.transition1[0](x[0]),
            self.transition1[1](x[-1]) + Recalibration_128
        ]

        x = self.stage2(x)
        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1]) + Recalibration_64
        ]

        x = self.stage3(x)
        x = [
            self.transition3[0](x[0]),
            self.transition3[1](x[1]),
            self.transition3[2](x[2]),
            self.transition3[3](x[-1]) + Recalibration_32
        ]

        x = self.stage4(x)

        res = x[0]
        x0_h, x0_w = x[0].size(2), x[0].size(3)

        SFM1 = self.SFM1(x[3], self.DA32(x[3]))
        SFM2 = self.SFM2(x[2], SFM1)
        SFM3 = self.SFM3(x[1], SFM2)
        SFM4 = self.SFM4(x[0], SFM3)

        out = torch.cat((F.interpolate(self.toplayer(SFM1), size=(x0_h, x0_w), mode='bilinear',
                                       align_corners=ALIGN_CORNERS),
                         F.interpolate(self.latlayer1(SFM2), size=(x0_h, x0_w), mode='bilinear',
                                       align_corners=ALIGN_CORNERS),
                         F.interpolate(self.latlayer2(SFM3), size=(x0_h, x0_w), mode='bilinear',
                                       align_corners=ALIGN_CORNERS),
                         SFM4,
                         res), dim=1)
        out = self.final_layer(out)
        out = F.interpolate(out, size=(x0_h * 4, x0_w * 4), mode='bilinear', align_corners=ALIGN_CORNERS)

        return out


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    rgb = torch.randn(1, 3, 1024, 1024).to(device)
    msi = torch.rand(1, 8, 256, 256).to(device)
    model = HighResolutionNet().to(device)
    outputs = model(rgb, msi)
    print(outputs.shape)
