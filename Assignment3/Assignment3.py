from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import os
import torch.nn.functional as F

#Convolutional neural network (one convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNet, self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(3,6,kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=2))
        self.layer2= nn.Sequential(
            nn.Conv2d(6,16,kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc_layer = nn.Sequential(
            nn.Linear(64*13*13, 4),
            nn.Linear(4, num_classes))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out=self.layer3(out)
        out = self.layer4(out)
        out = F.dropout(out, training=self.training)
        out=out.reshape(out.size(0),-1)
        out=self.fc_layer(out)
        return out

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = './hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

inputs, classes = next(iter(dataloaders['train']))

out = torchvision.utils.make_grid(inputs)

model=ConvNet(2).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

num_epochs=10
num_classes=2
batch_size=64
learning_rate=0.001

#Loss and Optimizer
criterion =nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print(ConvNet)

#Train the model
for epoch in range(num_epochs):

    running_loss=0.0
    running_corrects=0

    for i, (images, labels) in enumerate(dataloaders['train']):
        images=images.to(device)
        labels=labels.to(device)

        #Forward pass
        outputs=model(images)
        loss=criterion(outputs, labels)
        _, preds=torch.max(outputs, 1)

        #Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()*images.size(0)
        running_corrects+=torch.sum(preds == labels.data)

        epoch_loss=running_loss/dataset_sizes['train']
        epoch_acc=running_corrects.double() / dataset_sizes['train']

        if(i+1)%60==0:

            print('Train Epoch [{}/{}], Loss {:.4f}, Acc: {:.4f}'.format(epoch+1, num_epochs, loss.item(), epoch_acc))

#Test the model
model.eval()
with torch.no_grad():
    correct =0
    total=0
    for i,(images, labels) in enumerate(dataloaders['val']):
        images=images.to(device)
        labels=labels.to(device)
        outputs=model(images)
        loss=criterion(outputs, labels)
        _, predicted=torch.max(outputs.data, 1)
        total += labels.size(0)
        correct+=(predicted == labels).sum().item()

    print('Val Loss : {:.4f} Acc :: {:.4f}'.format(loss.item(), 100*correct/total))


