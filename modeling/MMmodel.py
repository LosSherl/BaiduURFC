import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels
class FCViewer(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

'''Dual Path Networks'''
class Bottleneck(nn.Module):
    def __init__(self, last_channels, in_channels, out_channels, dense_depth, stride, first_layer):
        super(Bottleneck, self).__init__()
        self.out_channels = out_channels
        self.dense_depth = dense_depth

        self.conv1 = nn.Conv2d(last_channels, in_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=32, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.conv3 = nn.Conv2d(in_channels, out_channels + dense_depth, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels + dense_depth)

        self.shortcut = nn.Sequential()
        if first_layer:
            self.shortcut = nn.Sequential(
                nn.Conv2d(last_channels, out_channels + dense_depth, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels + dense_depth)
            )
        
        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            x = self.shortcut(x)
            d = self.out_channels
            out = torch.cat([x[:,:d,:,:] + out[:,:d,:,:], x[:,:d,:,:], out[:,d:,:,:]],1)
            out = F.relu(out)
            return out

class DPN(nn.Module):
    def __init__(self, cfg):
        super(DPN, self).__init__()
        in_channels, out_channels = cfg['in_channels'], cfg['out_channels']
        num_blocks, dense_depth = cfg['num_blocks'], cfg['dense_depth']

        self.conv1 = nn.Conv2d(7, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.last_channels = 64
        self.layer1 = self._make_layer(in_channels[0], out_channels[0], num_blocks[0], dense_depth[0], stride=1)
        self.layer2 = self._make_layer(in_channels[1], out_channels[1], num_blocks[1], dense_depth[1], stride=2)
        self.layer3 = self._make_layer(in_channels[2], out_channels[2], num_blocks[2], dense_depth[2], stride=2)
        self.layer4 = self._make_layer(in_channels[3], out_channels[3], num_blocks[3], dense_depth[3], stride=2)
        self.linear = nn.Linear(out_channels[3] + (num_blocks[3] + 1) * dense_depth[3], 64)

    def _make_layer(self, in_channels, out_channels, num_blocks, dense_depth, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(Bottleneck(self.last_channels, in_channels, out_channels, dense_depth, stride, i == 0))
            self.last_channels = out_channels + (i + 2) * dense_depth
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def DPN26():
    cfg = {
        'in_channels': (96, 192, 384, 768),
        'out_channels': (256, 512, 1024, 2048),
        'num_blocks': (16, 32, 24, 128),
        'dense_depth': (16,32,24,128)
    }
    return DPN(cfg)

def DPN92():
    cfg = {
        'in_channels': (96,192,384,768),
        'out_channels': (256,512,1024,2048),
        'num_blocks': (3,4,20,3),
        'dense_depth': (16,32,24,128)
    }
    return DPN(cfg)

class MultiModalNet(nn.Module):
    def __init__(self, backbone1, backbone2, drop, num_classes, pretrained=True):
        super().__init__()
        if pretrained:
            img_model = pretrainedmodels.__dict__[backbone1](num_classes=1000, pretrained="imagenet")
        else:
            img_model = pretrainedmodels.__dict__[backbone1](num_classes=1000, pretrained=None)
        
        self.visit_model = DPN26()

        self.img_encoder = list(img_model.children())[:-2]
        self.img_encoder.append(nn.AdaptiveAvgPool2d(1))
        self.img_encoder == nn.Sequential(*self.img_encoder)

        if drop > 0:
            self.img_fc = nn.Sequential(
                FCViewer(),
                nn.Dropout(drop),
                nn.Linear(img_model.last_linear.in_features, 256)
            )
        else:
            self.img_fc = nn.Sequential(
                FCViewer(),
                nn.Linear(img_model.last_linear.in_features, 256)
            )
        self.cls = nn.Linear(320, num_classes)
    
    def forward(self, x_img, x_vis):
        x_img = self.img_encoder(x_img)
        x_img = self.img_fc(x_img)

        x_vis = self.visit_model(x_vis)
        
        x_cat = torch.cat((x_img,x_vis), 1)
        x_cat = self.cls(x_cat)
        return x_cat