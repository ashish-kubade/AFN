import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch

device = torch.device("cuda")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #layers conv1_1 to conv4_3 will be using pretrained weights for ortho branch
        self.conv1_1 = nn.Conv2d(3, 64, 3, stride=1,padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, stride=1,padding=1)
        
        self.conv2_1 = nn.Conv2d(64, 128, 3, stride=1,padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, stride=1,padding=1)
        
        self.conv3_1 = nn.Conv2d(128, 256, 3, stride=1,padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, stride=1,padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, stride=1,padding=1)


        self.conv4_1 = nn.Conv2d(256, 512, 3, stride=1,padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, stride=1,padding=1) 
        self.conv4_3 = nn.Conv2d(512, 512, 3, stride=1,padding=1) 
        
        self.pool = nn.MaxPool2d(2, 2)
        self.out = nn.Conv2d(512, 64, 1, stride=1, padding=1)
  
    def forward(self, ortho_img_tensors):
        #print(x.shape)
        #print(y.shape)
        # branch1 
        x = self.pool(F.relu(self.conv1_2(F.relu(self.conv1_1(ortho_img_tensors))))) # Otrho channel, RGB input, 64 channels
        #print(x.shape)
        x1= F.relu(self.conv2_2(F.relu(self.conv2_1(x)))) # 128 channels, first skip connection
        #print(x1.shape)
        
        x = self.pool(x1)
        x2= F.relu(self.conv3_3(F.relu(self.conv3_2(F.relu(self.conv3_1(x)))))) # 256 channels, second skip connection
        
        x = self.pool(x2)
        x3= F.relu(self.conv4_3(F.relu(self.conv4_2(F.relu(self.conv4_1(x)))))) #512 channels, third skip connection
#        out = F.relu(self.out(x3))
        out = x3
        return out

if __name__=='__main__':
    model = Net()
