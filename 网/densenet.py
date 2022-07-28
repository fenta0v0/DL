# DenseNet CNN网络框架练习
import torch.nn
import torch.nn.functional as F


class DenseLayer(torch.nn.Module):
    def __init__(self, input_features_num, grow_rate, bn_size, drop_rate):
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.BatchNorm2d(input_features_num),
            torch.nn.ReLU(inplace=True),  # inplace直接返回修改值，效率更好
            torch.nn.Conv2d(in_channels=input_features_num, out_channels=bn_size * grow_rate, kernel_size=1,
                            stride=1, bias=False),
            torch.nn.BatchNorm2d(bn_size * grow_rate),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=bn_size * grow_rate, out_channels=grow_rate, padding=1, kernel_size=3, stride=1,
                            bias=False)
        )
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.layer(x)
        if self.drop_rate > 0:
            out = F.dropout(input=out, p=self.drop_rate, training=self.training)
        return torch.cat([out, x], 1)


class DenseBlock(torch.nn.Module):
    def __init__(self, num_layers, input_features_num, grow_rate, bn_size, drop_rate):
        super().__init__()
        self.block_layer = torch.nn.Sequential()
        for i in range(num_layers):
            layer = DenseLayer(input_features_num + i * grow_rate, grow_rate, bn_size, drop_rate)
            self.block_layer.add_module("DenseLayer%d" % (i + 1), layer)

    def forward(self, x):
        return self.block_layer(x)


class Transition(torch.nn.Module):
    def __init__(self, input_features_num, output_features_num):
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.BatchNorm2d(input_features_num),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=input_features_num, out_channels=output_features_num, kernel_size=1, stride=1,
                            bias=False),
            torch.nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.layer(x)


class DenseNet(torch.nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64,
                 bn_size=4, compression_rate=0.5, drop_rate=0, num_classes=10):
        super().__init__()
        # 这是首层
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=num_init_features, kernel_size=7, stride=2, bias=False,
                            padding=3),   # kernel_size=7 ,padding=3  >>>>>HW不变  bias：偏置，padding处理时bias可以根据偏置进行填充
            torch.nn.BatchNorm2d(num_features=num_init_features),    # 归一化，收敛更容易，使其满足均值为0，方差为1的规律
            torch.nn.ReLU(inplace=True),  # 激活函数
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)   # 最大池化
        )
        # 中间的Dense层,输入通道数64
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers, input_features_num=num_features,
                               grow_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
            self.features.add_module("DenseBlock%d" % (i + 1), block)
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                transition = Transition(input_features_num=num_features,
                                        output_features_num=int(compression_rate * num_features))
                self.features.add_module("Transition%d" % (i + 1), transition)
                num_features = int(num_features * compression_rate)
        self.features.add_module("BN_last", torch.nn.BatchNorm2d(num_features))
        self.features.add_module("Relu_last", torch.nn.ReLU(inplace=True))
        self.features.add_module("AvgPool2d", torch.nn.AvgPool2d(kernel_size=7, stride=1))
        self.classifier = torch.nn.Linear(num_features, num_classes)

        # params initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.bias, 0)
                torch.nn.init.constant_(m.weight, 1)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


if __name__ == '__main__':
    x = torch.randn(1, 1, 224, 224)
    net = DenseNet()
    print(net(x).shape)
