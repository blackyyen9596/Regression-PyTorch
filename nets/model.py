import torch
import torch.nn as nn
from torchsummary import summary
from tensorboardX import SummaryWriter


# ,nn.BatchNorm1d()
class Net(nn.Module):
    def __init__(self, features):
        super(Net, self).__init__()
        self.apply(self.init_weights)  # 初始化權重
        self.dropout = torch.nn.Dropout(0.05)
        self.layer1 = nn.Sequential(nn.Linear(features, 32))
        self.layer2 = nn.Sequential(nn.Linear(32, 16))
        self.layer3 = nn.Sequential(nn.Linear(16, 8))
        self.layer4 = nn.Sequential(nn.Linear(8, 4))
        self.layer5 = nn.Sequential(nn.Linear(4, 2))
        self.layer6 = nn.Sequential(nn.Linear(2, 1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


# 顯示模型架構
# model = Net(5).cuda()
# x = torch.rand(1, 5).cuda()
# summary(model, input_size=(1, 5))

# 將模型上傳至tensorboard
# with SummaryWriter(comment='Net') as w:
#     w.add_graph(model, x)