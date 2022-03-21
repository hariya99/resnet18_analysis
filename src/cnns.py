import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1, kernel_sizes=[3,1]):
        super(BasicBlock, self).__init__()
        kernel_b = kernel_sizes[0]
        kernel_s = kernel_sizes[1]
        # formula for padding
        pad_b = 1 + (kernel_b - 3) // 2
        pad_s = 1 + (kernel_s - 3) // 2

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=kernel_b, stride=stride, padding=pad_b, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_b,
                               stride=1, padding=pad_b, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=kernel_s, stride=stride, padding=pad_s, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class ResNet(nn.Module):
    def __init__(self, block, num_blocks, out_channels, kernel_sizes,
                num_classes=10, fourth_layer=True):
        super(ResNet, self).__init__()
        self.pool_kernel_sz = self.pool_kernel_sz = 32//(2**(len(out_channels) - 1))
        self.fourth_layer = fourth_layer
        self.in_planes = out_channels[0]

        self.conv1 = nn.Conv2d(3, out_channels[0], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels[0])
        self.layer1 = self._make_layer(block, out_channels[0], num_blocks[0], kernel_sizes, stride=1)
        self.layer2 = self._make_layer(block, out_channels[1], num_blocks[1], kernel_sizes, stride=2)
        self.layer3 = self._make_layer(block, out_channels[2], num_blocks[2], kernel_sizes, stride=2)
        if self.fourth_layer:
            self.layer4 = self._make_layer(block, out_channels[3], num_blocks[3], kernel_sizes, stride=2)
        self.linear = nn.Linear(out_channels[-1], num_classes)

    def _make_layer(self, block, planes, num_blocks, kernel_sizes, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, kernel_sizes))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        if self.fourth_layer:
            out = self.layer4(out)
        out = F.avg_pool2d(out, self.pool_kernel_sz)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18(blocks_list, out_channels, kernel_sizes, fourth_layer=True):
    # number of blocks in each layer
    # blocks_list = [2, 2, 2, 2]
    return ResNet(BasicBlock, blocks_list, out_channels, kernel_sizes, fourth_layer=fourth_layer)

if __name__ == "__main__":

    from torchinfo import summary
    # model = ResNet18([2,2,2,2], [64,128,256,512])
    model = ResNet18([1,1,1], [128,256,512], [3,1], fourth_layer=False)
    summary(model, input_size=(128, 3, 32, 32))