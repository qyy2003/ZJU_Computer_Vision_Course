import pandas as pd
import os
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

from PIL import Image
import copy
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
# from MobileNet import MobileNet


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# device="cpu"


## prepare data
df = pd.read_csv('pokemon.csv')

labels0 = df['Type1'].tolist()
labels0.extend(df['Type2'].tolist())
labels=list(set(labels0))
labels_num=len(labels)
dict2str={}
dict2int={}
en_weight=torch.zeros(labels_num,device=device)
for i,x in enumerate(labels):
    dict2str[i]=x
    dict2int[x]=i
    en_weight[i]=1.0/labels0.count(x)
    # en_weight[i]=1.0
en_weight=en_weight/en_weight.sum()*labels_num
print(en_weight)
print(dict2int)
NONE=0
while(not pd.isna(dict2str[NONE])):
    NONE+=1
print(NONE)
# en_weight[NONE]=0
dict_all={}
# read df line by line
for index, row in df.iterrows():
    # print(row['Type1'],row['Type2'])
    # print(dict2int[row['Type1']],dict2int[row['Type2']])
    dict_all[row['Name']]=[dict2int[row['Type1']],dict2int[row['Type2']]]
print(dict_all)

df0 = pd.read_csv('train.csv',header=None)
train_dataset=[]
for index, row in df0.iterrows():
    train_dataset.append(row[0])

df0 = pd.read_csv('test.csv',header=None)
test_dataset=[]
for index, row in df0.iterrows():
    test_dataset.append(row[0])
print("The total number is:",len(df)," ; The train number is:",len(train_dataset)," ; The test number is:",len(test_dataset))

## create torch dataset
class PokemonImageDataset(Dataset):
    def __init__(self, data,img_dir,label_dict,is_transform=None,repeat=1):
        self.repeat=repeat
        self.data=data
        self.img_dir = img_dir
        self.label_dict=label_dict
        height=width=120
        self.transform =T.Compose([
            # T.Resize([232]),
            T.RandomRotation(90,center=(height/2,width/2)),
            T.RandomResizedCrop((height, width),scale = (0.7,1.0)),
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            # T.CenterCrop([224]),
            # T.Random
            # T.RandomHorizontalFlip(0.1),  # 进行随机水平翻转
            # T.RandomVerticalFlip(0.1),  # 进行随机竖直翻转
            T.ToTensor(),  # 转化为张量
            # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 归一化
        ]) if is_transform else T.Compose([
            T.Resize([120]),
            # T.RandomRotation(15,center=(height/2,width/2)),
            # T.RandomResizedCrop((height, width),scale = (0.7,1.0)),
            # T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            # T.CenterCrop([224]),
            # T.Random
            # T.RandomHorizontalFlip(0.1),  # 进行随机水平翻转
            # T.RandomVerticalFlip(0.1),  # 进行随机竖直翻转
            T.ToTensor(),  # 转化为张量
            # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 归一化
        ])

    def __len__(self):
        return len(self.data)*self.repeat

    def __getitem__(self, idx):
        idx=idx%len(self.data)
        img_path = os.path.join(self.img_dir, self.data[idx]+'.png')
        if(not os.path.isfile(img_path)):
            img_path = os.path.join(self.img_dir, self.data[idx]+'.jpg')
            # print("!")

        # image=read_image(img_path)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # print(image.size(0))
        # if(image.size(0)==4):
        #     print(self.data[idx])
        #     image=image[:3,:,:]
        label0 = self.label_dict[self.data[idx]]
        # print(label0)
        label=torch.zeros(labels_num)
        label[label0[0]]=1
        label[label0[1]]=1
        # label[NONE]=0
        # if self.transform:
        #     image = self.transform(image)
        return image, label


class myNet(nn.Module):
    def __init__(self, classes=2):
        super(myNet, self).__init__()

        # self.model1= nn.Sequential(
        #     nn.Conv2d(3, 16, (3, 3)),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.MaxPool2d((2, 2)),
        #
        #     nn.Conv2d(16, 32, (3, 3)),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.MaxPool2d((2, 2)),
        #
        #     nn.Conv2d(32, 64, (3, 3)),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.MaxPool2d((2, 2)),
        #
        #     nn.Conv2d(64, 128, (3, 3)),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.MaxPool2d((2, 2)),
        #
        #     nn.Conv2d(128, 150, (3, 3)),
        #     nn.BatchNorm2d(150),
        #     nn.ReLU(),
        #     nn.MaxPool2d((2, 2)),
        #
        #     nn.Flatten(),
        #     nn.Linear(150,64),
        #     nn.ReLU(),
        #     nn.Linear(64,classes),
        #     nn.Softmax()
        # )
        self.model2= nn.Sequential(
            nn.Conv2d(3, 16, (3, 3)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((4, 4)),

            nn.Conv2d(16, 64, (3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((4, 4)),

            nn.Conv2d(64, 64, (3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d((4, 4)),

            nn.Flatten(),
            nn.Linear(1024,classes),
            nn.Softmax()
        )
        # self.model3= nn.Sequential(
        #     nn.Conv2d(3, 8, (3, 3)),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(),
        #     nn.MaxPool2d((4, 4)),
        #
        #     nn.Conv2d(8, 16, (3, 3)),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.MaxPool2d((4, 4)),
        #
        #     nn.Conv2d(16, 32, (3, 3)),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #
        #     nn.Flatten(),
        #     nn.Linear(512,classes),
        #     nn.Softmax()
        # )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.1)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.model2(x)
        return x


batch_size=8

training_data= PokemonImageDataset(train_dataset,img_dir='images/',label_dict=dict_all,is_transform=True,repeat=3)
test_data= PokemonImageDataset(test_dataset,img_dir='images/',label_dict=dict_all,is_transform=False,repeat=1)
train_data_loader = DataLoader(training_data, batch_size=batch_size,shuffle=True)
valid_data_loader = DataLoader(test_data, batch_size=batch_size,shuffle=True)


epochs = 100
model = myNet(classes=len(labels)).to(device)
# model.load_state_dict(
#                 torch.load('./results/temp1.pth', map_location=device))
# optimizer = optim.Adam(model.parameters(), lr=0.05)  # 优化器
optimizer =optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
print('加载完成...')

milestones=[10,25,50,75]
# 学习率下降的方式，acc三次不下降就下降学习率继续训练，衰减学习率
scheduler =torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.5, last_epoch=-1, verbose=False)
# 损失函数
criterion = nn.functional.binary_cross_entropy

best_loss = 1e9
best_acc=0
best_acc1=0
best_model_weights = copy.deepcopy(model.state_dict())

for epoch in range(epochs):
    model.train()
    total_loss= 0
    total_acc=0
    total_num=0
    total_acc1=0
    total_num1=0
    for batch_idx, (x, y) in tqdm(enumerate(train_data_loader, 1)):
        x = x.to(device)
        # print(y.size())
        y = y.to(device)
        pred_y = model(x)

        # print(pred_y)
        # print(y.shape)

        loss = criterion(pred_y, y,weight=en_weight)
        pred_y = torch.sort(pred_y, dim=1,descending=True)[1]
        for y1,y2 in zip(pred_y,y):
            if y1[1]==NONE:
                total_acc1+=y2[y1[0]]
            elif y1[0]==NONE:
                total_acc1+=y2[y1[1]]
            else:
                total_acc1+=y2[y1[0]]*0.5+y2[y1[1]]*0.5

        total_num1+=x.shape[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+= loss.item()
        # if(batch_idx==1):
        #     print('step:' + str(batch_idx) + '/' + str(len(train_data_loader)) + ' || Loss: %.4f' % (loss)+ ' || Train Acc: %.4f' % ((pred_y==y).sum()))
        # print('step:' + str(batch_idx) + '/' + str(len(train_data_loader)) + ' || Total Loss: %.4f' % (total_loss/total_num1)+ ' || Train Acc: %.4f' % (total_acc1/total_num1))
    scheduler.step()
    writer.add_scalar("Loss/train", total_loss/total_num1, epoch)
    writer.add_scalar("Acc/train", total_acc1/total_num1, epoch)
    print(optimizer.state_dict()['param_groups'][0]['lr'])
    model.eval()
    acc_loss=0
    for batch_idx, (x, y) in tqdm(enumerate(valid_data_loader, 1)):
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            pred_y = model(x)
            loss1 = criterion(pred_y, y)
            acc_loss+= loss1.item()
        # get the predicted labels of pred_y

        pred_y = torch.sort(pred_y, dim=1,descending=True)[1]
        for y1,y2 in zip(pred_y,y):
            if y1[1]==NONE:
                total_acc+=y2[y1[0]]
            elif y1[0]==NONE:
                total_acc+=y2[y1[1]]
            else:
                total_acc+=y2[y1[0]]*0.5+y2[y1[1]]*0.5
            # total_acc += y2[y1[0]] * 0.5 + y2[y1[1]] * 0.5
        # pred_y = torch.max(pred_y, dim=1)[1]
        # total_acc+= (pred_y==y).sum()
        total_num+=x.shape[0]

    if total_acc > best_acc or (total_acc == best_acc and total_acc1>best_acc1):
        best_model_weights = copy.deepcopy(model.state_dict())
        best_acc = total_acc
    writer.add_scalar("Loss/validation", acc_loss/total_num, epoch)
    writer.add_scalar("Acc/validation",  total_acc/total_num, epoch)
    print('step:' + str(epoch + 1) + '/' + str(epochs) + ' || Total Loss: %.4f' % (total_loss/total_num1)+ ' || Train Acc: %.4f' % (total_acc1/total_num1)+' || Valid Acc: %.4f' % (total_acc/total_num))

    if(epoch%10==0):
        torch.save(best_model_weights, './results/temp.pth')
        # print('Finish Training.')
torch.save(best_model_weights, './results/temp.pth')
print('Finish Training.')