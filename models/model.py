import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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




class YouOwnModel(nn.Module):
    # set you own model structure here.
    def __init__(self, feature_dimension = 512, block = Bottleneck, num_classes=500):
        pass

    def forward(self, silho):
        pass



class SilhouetteDeep(nn.Module):

    def __init__(self, feature_dimension = 512 ,block = Bottleneck, num_classes=86):
        print("num_classes:", num_classes)
        self.inplanes = 64
        super(SilhouetteDeep, self).__init__()
        self.conv1 = nn.Sequential(conv3x3(1,64),
                                   nn.BatchNorm2d(64),
                                   conv3x3(64, 64),
                                   nn.BatchNorm2d(64),
                                   nn.MaxPool2d(2,2),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2, stride=2)
                                   )

        self.layer1 = BasicBlock(64, 64)
        self.conv2 = nn.Sequential(conv3x3(64,128),
                                   nn.MaxPool2d(2, 2))

        self.layer2 = nn.Sequential(BasicBlock(128, 128),
                                    BasicBlock(128, 128))
        self.conv3 = nn.Sequential(conv3x3(128,256),
                                   nn.MaxPool2d(2, 2))
        self.layer3 =  nn.Sequential(Bottleneck(256, 64),
                                    Bottleneck(256, 64),
                                    Bottleneck(256, 64))
        self.conv4 = nn.Sequential(conv3x3(256,512),
                                   nn.MaxPool2d(2, 2))
        self.layer4 = nn.Sequential(Bottleneck(512, 128),
                                    Bottleneck(512, 128),
                                    Bottleneck(512, 128))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512,feature_dimension)
        self.out = nn.Linear(feature_dimension, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, silho):
        n = silho.size(1)
        print("n : =", n)

        out = []
        for i in range(n):
            input = silho[:,i,:,:].unsqueeze(1)
            print("input size",input.size())
            x = self.conv1(input)
            x = self.layer1(x)
            x = self.conv2(x)
            x = self.layer2(x)
            x = self.conv3(x)
            x = self.layer3(x)

            x = self.conv4(x)
            x = self.layer4(x)

            x = self.layer4(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            temp = x.unsqueeze(1)

            if i !=0:
                out = torch.cat((out, temp), 1)
            else:
                out = temp

        fc = torch.mean(out, 1)
        out = self.out(fc)

        return fc, out


class SilhouetteNormal(nn.Module):

    def __init__(self, feature_dimension = 512, block = Bottleneck, num_classes=86):
        print("num_classes:", num_classes)
        self.inplanes = 64
        super(SilhouetteNormal, self).__init__()
        self.conv1 = nn.Sequential(conv3x3(1,64),
                                   nn.BatchNorm2d(64),
                                   conv3x3(64, 64),
                                   nn.BatchNorm2d(64),
                                   nn.MaxPool2d(2,2),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2, stride=2)
                                   )

        self.layer1 = BasicBlock(64, 64)
        self.conv2 = nn.Sequential(conv3x3(64,128),
                                   nn.MaxPool2d(2, 2))

        self.layer2 = nn.Sequential(BasicBlock(128, 128))
        self.conv3 = nn.Sequential(conv3x3(128,256),
                                   nn.MaxPool2d(2, 2))
        self.layer3 =  nn.Sequential(Bottleneck(256, 64))
        self.conv4 = nn.Sequential(conv3x3(256,512),
                                   nn.MaxPool2d(2, 2))
        self.layer4 = nn.Sequential(Bottleneck(512, 128))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512,feature_dimension)
        self.out = nn.Linear(feature_dimension, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, silho):
        n = silho.size(1)  #batch
        out = []
        for i in range(n):
            input = silho[:,i,:,:].unsqueeze(1)
            x = self.conv1(input)
            x = self.layer1(x)
            x = self.conv2(x)
            x = self.layer2(x)
            x = self.conv3(x)
            x = self.layer3(x)

            x = self.conv4(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            temp = x.unsqueeze(1)
            if i !=0:
                out = torch.cat((out, temp), 1)
            else:
                out = temp
        fc = torch.mean(out, 1)
        out = self.out(fc)

        return fc, out


if __name__=="__main__":
    model = SilhouetteNormal()
    print(model)