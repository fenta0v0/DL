import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 2)

    def forward(self, x):
        return self.conv1(x)


model = CNN()
def show_lr(epochs,schduler):
    lr_list = []
    for i in range(epochs):
        lr = schduler.get_last_lr()
        lr_list.append(lr)

        optimx.step()
        schduler.step()

    plt.plot(lr_list)
    plt.show()
    for i ,lr in enumerate(lr_list):
        print(f"Epoch: {i}, LR:{lr}")


optimx = torch.optim.SGD(model.parameters(),lr=0.1)
schoduler = torch.optim.lr_scheduler.StepLR(optimx,step_size=10,gamma=0.5)
show_lr(100,schoduler)
"""
step:10
LR: 0.1-> 0.1*0.5 ->0.1*0.5*0.5
"""

optimx = torch.optim.SGD(model.parameters(),lr=0.1)
schoduler = torch.optim.lr_scheduler.MultiStepLR(optimx,milestones=[10,30,40],gamma=0.5)
show_lr(100,schoduler)

"""
epoch:0-10: 0.1
epoch:11-30: 0.1*0.5
epoch:31-40: 0.1*0.5*0.5
epoch:41-100:0.1*0.5*0.5*0.5
"""

optimx = torch.optim.SGD(model.parameters(),lr=0.1)
schoduler = torch.optim.lr_scheduler.ConstantLR(optimx,factor=0.333,total_iters=20)
show_lr(100,schoduler)
"""
分为两部分：
0-20： 0.1*0.333
20-100： 0.1
"""

optimx = torch.optim.SGD(model.parameters(),lr=0.1)
schoduler = torch.optim.lr_scheduler.LinearLR(optimx,start_factor=0.333,end_factor=1.0,total_iters=50)
show_lr(100,schoduler)
"""
分为两部分：
0-50:上升 0.1*0.333 ~0.1
50-100： 不变 0.1
"""
optimx = torch.optim.SGD(model.parameters(),lr=0.1)
schoduler = torch.optim.lr_scheduler.LinearLR(optimx,start_factor=1,end_factor=0.1,total_iters=50)
show_lr(100,schoduler)
"""
分为两部分：
0-50:下降 1 ~0.1
50-100： 不变 0.1
"""

optimx = torch.optim.SGD(model.parameters(),lr=0.1)
schoduler = torch.optim.lr_scheduler.ExponentialLR(optimx,gamma=0.95)
show_lr(100,schoduler)
"""
每个step 都要上一步的LR*gamma
"""

optimx = torch.optim.SGD(model.parameters(),lr=0.1)
schoduler = torch.optim.lr_scheduler.LambdaLR(optimx,lr_lambda=lambda epoch:1/(epoch+10))
show_lr(100,schoduler)
"""
自定义LR：
"""

optimx = torch.optim.SGD(model.parameters(),lr=0.1)
schoduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimx,T_max=20,eta_min=0.001)
show_lr(100,schoduler)
"""
cosine 循环
"""
optimx = torch.optim.SGD(model.parameters(),lr=0.1)
schoduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimx,T_0=20,T_mult=2,eta_min=0.001)
show_lr(100,schoduler)
"""
cosine变体
"""

optimx = torch.optim.SGD(model.parameters(),lr=0.1)
schoduler = torch.optim.lr_scheduler.CyclicLR(optimx,base_lr=0.001,max_lr=0.01,step_size_up=10,step_size_down=20,mode='triangular')
show_lr(100,schoduler)
"""
每个周期最高减半
"""
