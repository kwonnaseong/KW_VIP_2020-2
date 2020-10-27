import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.models as models

class FeatureExtraction(torch.nn.Module):
    def __Init__(self):
        super(FeatureExtraction. self).__init__()

        self.model=models.vgg16(pretrained=True)

        last_layer_idx=23
        self.model= nn.Sequential(*list(self.model.features.children())[:last_layer_idx+1])

        for param in self.model.parameters():
            param.requires_grad=False

    def forward(self, image_batch):
        return self.mode(image_batch)

#net = FeatureExtraction()
net = models.vgg16(pretrained=True)
print(net)

conv_layer=[]
for cnt in range(len(net.features)):
    for param in net.features[cnt].parameters():
        conv_layer.append(cnt)

conv_layer= list(set(conv_layer))
print(conv_layer)

clf_layer=[]
for cnt in range(len(net.classifier)):
    for param in net.classifier[cnt].parameters():
        clf_layer.append(cnt)
    clf_layer=list(set(clf_layer))
print(clf_layer)

for cnt in range(conv_layer[5]+1):
    for param in net.featrues[cnt].parameters():
        param.requires_grad=False


def forward(self, image_batch):
    return self.model(image_batch)

class FeatureRegression(nn.Module):
    def __init__(self):
        super(FeatureRegression, self).__init__()
        fc_input = 32*5*5
        self.linear = nn.Linear(fc_input, 2)

    def forward(self, x):

        x= x.view(x.size(0), -1)
        x= self.linear(x)
        return x

class MyTransferNet(nn.Module):
    def __init__(self):
        super(MyTransferNet, self).__init__()

        self.FeatureExtraction = FeatureExtraction()
        self.FeatureExtraction = FeatureRegression()

    def forward(self, tnf_batch):
        output= self.FeatureExtraction(tnf_batch)

        output = self.FeatureRegression(output)

        return output

net = MyTransferNet()

optimizer = torch.optim.SGD(net.parameters(, lr = 0.01))

