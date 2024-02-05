import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__() # 利用参数初始化父类
        self.backbone= nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.classifier=nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x=self.backbone(x)
        x=x.view(x.size(0), -1)
        x=self.classifier(x)
        return x


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = LeNet().to(device)
    model=model.to(device)
    print(model)
    # MNIST
    train_dataset=torchvision.datasets.MNIST("mnist", train=True, transform=torchvision.transforms.ToTensor(), target_transform=None, download=True)
    test_dataset=torchvision.datasets.MNIST("mnist", train=False, transform=torchvision.transforms.ToTensor(), target_transform=None, download=True)
    print(len(train_dataset), len(test_dataset))
    train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader=torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
    # optimizer
    optimizer=torch.optim.Adam(model.parameters(), lr=0.01)
    # optimizer=torch.optim.SGD(model.parameters(), lr=0.01,momentum=0.8)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) #gamma=0.5
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9)

    # loss
    criterion=nn.CrossEntropyLoss()
    for epoch in range(50):
        #train
        model.train()
        correct=0
        train_loss=0
        total=0
        for i, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            data=data.to(device)
            # print(data.sum())
            target=target.to(device)
            output=model(data)
            loss=criterion(output, target)
            train_loss+=loss.item()
            _, pred= torch.max(output.data, 1)
            total+=target.size(0)
            correct+=(pred==target).sum().item()
            loss.backward()
            optimizer.step()
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Acc/train", correct/total, epoch)
        print("Train Epoch",epoch," | total loss=",train_loss,"acc=", correct/total*100,"%")
        # test
        # scheduler.step()
        test_loss=0
        correct=0
        total=0
        model.eval()
        with torch.no_grad():
            for data, target in test_loader:
                data=data.to(device)
                target=target.to(device)
                output=model(data)
                test_loss+=criterion(output, target).item()
                total += target.size(0)
                _, pred= torch.max(output.data, 1)
                # print(output.data, pred,target)
                correct+=(pred==target).sum().item()
        writer.add_scalar("Loss/validation", test_loss, epoch)
        writer.add_scalar("Acc/validation",  correct/total, epoch)
        print("Eval Epoch",epoch,"| total loss=",test_loss,"acc=", correct/total*100,"%")

    writer.flush()
    writer.close()
