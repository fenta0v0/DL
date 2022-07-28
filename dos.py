import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

"""
os模块用法
1、获取当前的工作路径：os.getcwd()
2、获取文件列表： 
    os.listdir():  直接返回指定路径下，文件和文件夹组成的列表
    os.walk():  传入一个path,获取每个层文件下的文件路径、文件夹列表、文件列表
3、判断某个文件夹是否存在：os.path.exists(): 若文件存在，则返回TRUE，否则返回FALSE
4、创建文件夹：
    os.mkdir() 创建一个新的文件夹
    os.makedirs() 递归生成文件夹
5、删除文件夹
    os.rmdir() 
6、路径拼接与切分
    os.path.join():路径拼接
    os.path.split():路径切分
7.单独获取文件的绝对路径、文件名
    os.path.dirname():返回文件的绝对路径
    os.path.basename():返回绝对路径下的“文件名”
8、判断是文件还是文件夹
    os.path.isdir():判断是否是文件夹
    os.path.isfile():判断是否是文件
9、os.step:返回当前操作系统的路径分隔符
os.path.getsize(文件)：获取文件大小
"""


"""
torch模块用法：
1、torch模块
import torch
包含了多维张量的数据结构以及基于其上的多种数学操作。另外，它也提供了多种工具，其中一些可以更有效地对张量和任意类型进行序列化。
具体包括pytorch张量的生成，以及运算、切片、连接等操作，还包括神经网络中经常使用的激活函数，比如sigmoid、relu、tanh，还提供了与numpy的交互操作
2、torch.Tensor
import torch
a = torch.tensor()
numpy作为Python中数据分析的专业第三方库，比Python自带的Math库速度更快。
同样的，在PyTorch中，有一个类似于numpy的库，称为Tensor。Tensor可谓是神经网络界的numpy
3、torch.sparse
在做nlp任务时，有个特点就是特征矩阵是稀疏矩阵。torch.sparse模块定义了稀疏张量，采用的是COO格式，
主要方法是用一个长整型定义非零元素的位置，用浮点数张量定义对应非零元素的值。稀疏张量之间可以做加减乘除和矩阵乘法。从而有效地存储和处理大多数元素为零的张量。
4、torch.cuda
该模块定义了与cuda运算的一系列函数，比如检查系统的cuda是否可用，在多GPU情况下，查看显示当前进程对应的GPU序号，清除GPU上的缓存，设置GPU的计算流，同步GPU上执行的所有核函数等。
5、torch.nn模块
torch.nn是pytorch神经网络模块化的核心，这个模块下面有很多子模块，
包括卷积层nn.ConvNd和线性层（全连接层）nn.Linear等。当构建深度学习模型的时候，可以通过继承nn.Module类并重写forward方法来实现一个新的神经网络。
另外，torch.nn中也定义了一系列的损失函数，包括平方损失函数torch.nn.MSELoss、交叉熵损失函数torch.nn.CrossEntropyLoss等。
6、torch.nn.functional函数模块
该模块定义了一些与神经网络相关的函数，包括卷积函数和池化函数等，torch.nn中定义的模块一般会调用torch.nn.functional里的函数，
比如，nn.ConvNd会调用torch.nn.functional.convNd函数。另外，torch.nn.functional里面还定义了一些不常用的激活函数，
包括torch.nn.functional.relu6和torch.nn.functional.elu等。
7、torch.nn.init模块
该模块定义了神经网络权重的初始化，包括均匀初始化torch.nn.init.uniform_和正太分布归一化torch.nn.init.normal_等。
值得注意得是，在pytorch中函数或者方法如果以下划线结尾，则这个方法会直接改变作用张量的值，因此，这些方法会直接改变传入张量的值，同时会返回改变后的张量。
8、torch.optim模块
torch.optim模块定义了一系列的优化器，比如torch.optim.SGD、torch.optim.AdaGrad、torch.optim.RMSProp、torch.optim.Adam等。
还包含学习率衰减的算法的模块torch.optim.lr_scheduler，这个模块包含了学习率阶梯下降算法torch.optim.lr_scheduler.StepLR
和余弦退火算法torch.optim.lr_scheduler.CosineAnnealingLR
9、torch.autograd模块
该模块是pytorch的自动微分算法模块，定义了一系列自动微分函数，包括torch.autograd.backward函数，主要用于在求得损失函数后进行反向梯度传播。
torch.autograd.grad函数用于一个标量张量（即只有一个分量的张量）对另一个张量求导，以及在代码中设置不参与求导的部分。
另外，这个模块还内置了数值梯度功能和检查自动微分引擎是否输出正确结果的功能。
10、torch.distributed模块
torch.distributed是pytorch的分布式计算模块，主要功能是提供pytorch的并行运行环境，其主要支持的后端有MPI、Gloo和NCCL三种。
pytorch的分布式工作原理主要是启动多个并行的进程，每个进程都拥有一个模型的备份，然后输入不同的训练数据到多个并行的进程，计算损失函数，每个进行独立地做反向传播，最后对所有进程权重张量的梯度做归约（Redue）。
用到后端的部分主要是数据的广播（Broadcast）和数据的收集（Gather），其中，前者是把数据从一个节点（进程）传播到另一个节点（进程），比如归约后梯度张量的传播，后者则把数据从其它节点转移到当前节点，比如把梯度张量从其它节点转移到某个特定的节点，然后对所有的张量求平均。
pytorch的分布式计算模块不但提供了后端的一个包装，还提供了一些启动方式来启动多个进程，包括但不限于通过网络（TCP）、环境变量、共享文件等。
11、torch.distributions模块
该模块提供了一系列类，使得pytorch能够对不同的分布进行采样，并且生成概率采样过程的计算图。
在一些应用过程中，比如强化学习，经常会使用一个深度学习模型来模拟在不同环境条件下采取的策略，其最后的输出是不同动作的概率。
当深度学习模型输出概率之后，需要根据概率对策略进行采样来模拟当前的策略概率分布，最后用梯度下降方法来让最优策略的概率最大（这个算法称为策略梯度算法，Policy Gradient）。
实际上，因为采样的输出结果是离散的，无法直接求导，所以不能使用反keh.distributions.Categorical类，pytorch还支持其它分布。
比如torch.distributions.Normal类支持连续的正太分布的采样，可以用于连续的强化学习的策略。
12、torch.hub模块
该模块提供了一系列预训练的模型供用户使用。比如，可以通过torch.hub.list函数来获取某个模型镜像站点的模型信息。
通过torch.hub.load来载入预训练的模型，载入后的模型可以保存到本地，并可以看到这些模型对应类支持的方法。
13、torch.jit模块
该模块是pytorch的即时编译器模块。这个模块存在的意义是把pytorch的动态图转换成可以优化和序列化的静态图，其主要工作原理是通过预先定义好的张量，追踪整个动态图的构建过程，得到最终构建出来的动态图，然后转换为静态图。
通过JIT得到的静态图可以被保存，并且被pytorch其它前端（如C++语言的前端）支持。另外，JIT也可以用来生成其它格式的神经网络描述文件，如ONNX。torch.jit支持两种模式，即脚本模式（ScriptModule）和追踪模式（Tracing）。
两者都能构建静态图，区别在于前者支持控制流，后者不支持，但是前者支持的神经网络模块比后者少。
14、torch.multiprocessing模块
该模块定义了pytorch中的多进程API，可以启动不同的进程，每个进程运行不同的深度学习模型，并且能够在进程间共享张量。
共享的张量可以在CPU上，也可以在GPU上，多进程API还提供了与python原生的多进程API（即multiprocessing库）相同的一系列函数，包括锁（Lock）和队列（Queue）等。
15、torch.random模块
该模块提供了一系列的方法来保存和设置随机数生成器的状态，包括使用get_rng_state函数获取当前随机数生成器的状态，set_rng_state函数设置当前随机数生成器状态，并且可以使用manual_seed函数来设置随机种子，也可以使用initial_seed函数来得到程序初始的随机种子。
因为神经网络的训练是一个随机的过程，包括数据的输入、权重的初始化都具有一定的随机性。设置一个统一的随机种子可以有效地帮助我们测试不同神经网络地表现，有助于调试神经网络地结构。
16、torch.onnx模块
该模块定义了pytorch导出和载入ONNX格式地深度学习模型描述文件。ONNX格式地存在是为了方便不同深度学习框架之间交换模型。
引入这个模块可以方便pytorch导出模型给其它深度学习框架使用，或者让pytorch载入其它深度学习框架构建地深度学习模型。
17、torch.utils模块
该模块提供了一系列地工具来帮助神经网络地训练、测试和结构优化。这个模块主要包含以下6个子模块：
1，torch.utils.bottleneck模块
该模块可以用来检查深度学习模型中模块地运行时间，从而可以找到性能瓶颈的那些模块，通过优化那些模块的运行时间，从而优化整个深度学习的模型的性能。
2，torch.utils.checkpoint模块
该模块可以用来节约深度学习使用的内存。通过前面的介绍我们知道，因为要进行梯度反向传播，在构建计算图的时候需要保存中间的数据，而这些数据大大增加了深度学习的内存消耗。
为了减少内存消耗，让迷你批次的大小得到提高，从而提升深度学习模型的性能和优化时的稳定性，我们可以通过这个模块记录中间数据的计算过程，然后丢弃这些中间数据，等需要用到的时候再重新计算这些数据。
这个模块设计的核心思想是以计算时间换内存空间，如果使用得当，深度学习模型的性能可以有很大的提升。
3，torch.utils.cpp_extension模块
该模块定义了pytorch的C++扩展，其主要包含两个类：CppExtension定义了使用C++来编写的扩展模块的源代码相关信息，CUDAExtension则定义了C++/CUDA编写的扩展模块的源代码相关信息。
再某些情况下，用户可能使用C++实现某些张量运算和神经网络结构（比如pytorch没有类似功能的模块或者类似功能的模块性能比较低），该模块就提供了一个方法能够让python来调用C++/CUDA编写的深度学习扩展模块。
在底层上，这个扩展模块使用了pybind11,保持了接口的轻量性并使得pytorch易于被扩展。
4，torch.utils.data模块
该模块引入了数据集（Dataset）和数据载入器（DataLoader）的概念，前者代表包含了所有数据的数据集，通过索引能够得到某一条特定的数据，后者通过对数据集的包装，可以对数据集进行随机排列（Shuffle）和采样（Sample）,得到一系列打乱数据的迷你批次。
5，torch.util.dlpacl模块
该模块定义了pytorch张量和DLPackz张量存储格式之间的转换，用于不同框架之间张量数据的交换。
6，torch.utils.tensorboard模块
该模块是pytorch对TensorBoard数据可视化工具的支持。TensorBoard原来是TensorFlow自带的数据可视化工具，能够显示深度学习模型在训练过程中损失函数、张量权重的直方图，以及模型训练过程中输出的文本、图像和视频等。
TensorBoard的功能非常强大，而且是基于可交互的动态网页设计的，使用者可以通过预先提供的一系列功能来输出特定的训练过程的细节（如某一神经网络层的权重的直方图，以及训练过程中某一段时间的损失函数等）pytorch支持TensorBoard可视化后，在训练过程中，可以很方便地观察中间输出地张量，也可以方便地调试深度学习模型。
"""


# 裁剪图片进行处理将图片转为256*256
def keep_image_size_open(path, size=(256, 256)):
    img = Image.open(path)
    temp = max(img.size)  # 最长边
    mask = Image.new('RGB', (temp, temp), (0, 0, 0))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask


# ===================Image==========================

"""
Image:
Image.new(model，size,color = 0)
model = 1 位图，像素1位
model = L 灰度图，像素8位
model = I 像素int32
model = F 像素float32
model = P 8位，映射为其他模式
model = RGB 真色彩，3通道
model = RGBA 4通道，加透明
model = CMYK 印刷，4通道
model = YCbCr 亮色分离，3通道

2.读取一张图片
im=Image.open( ' /home.picture/test.jpg' )

3.显示一张图片
im.show()

4.保存一张图片
im.save( 'save.gif ' , GIF)#把图像保存为gif的格式

5.创建新图片：
Image.new(model,size)
Image.new(model,size,color)
举个例子：
Newlmg =Image.new('RGBA',(640,480),(0,255,0))
newimg.save('newimg.png','PNG')

6.两张图片相加
Image.blend(img1,img2,alpha)#其中alpha指的是img1和img2相加的比例参数

7.点操作
Im.point(function)#对图像中的每个点执行函数function
举个例子：out=im.point(lambda i:i*1.5)#对于图片中的像素进行1.5倍的加强。（对于lambda函数输入是i，而输出是i*1.5）

8.查看图像信息
im.format,im.size.im.mode

9.图片裁剪
box=(100,100,500,500)
设置要裁剪的区域
region=im.crop(box)#region是一个新图像的对象

10.图像黏贴（合并）
im.paste(region,box)#黏贴box大小的region到原先图片中。

11.同道分离
r,g,b=im.split()#分割成三个R，G，B通道，次时的r，g，b分别为三个图像的对象。

12.合并通道
im=Image.merge('RGB',(b,g,r))#将b，r两个通道进行翻转

13.改变图像的的大小
out=im.resize((128,128))

14.图像翻转
out=img.rotate(45)

15图像转换
左右转换：out=im.transpose(Image.FLIP_LEFT_RIGHT)
上下对换：out=im.transpose(Image.FLIP_TOP_BOTTOM)

16.图像；类型的转换：
im=im.convert('RGBA')

17.获取某个像素位置的值：
im.getx((4,4))

18.写某个像素的值

im.put pixel((4,4),(255,0,0))
"""


# ================transforms================
"""
transforms.Compose() 为将几个变换组合在一起。这个转换不支持torch-script
transforms.ToTensor() 为转换一个PIL库的图片或者numpy的数组为tensor张量类型；转换从[0,255]->[0,1]
transforms.Normalize() 为通过平均值和标准差来标准化一个tensor图像，公式为：output[channel] = (input[channel] - mean[channel]) / std[channel]
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))解释：第一个(0.5,0.5,0.5) 即三个通道的平均值 第二个(0.5,0.5,0.5) 即三个通道的标准差值
由于ToTensor()已经将图像变为[0,1]，我们使其变为[-1,1]，以第一个通道为例，将最大与最小值代入公式：(0-0.5)/0.5=-1 ，(1-0.5)/0.5=1 其他数值同理操作，即映射到[-1,1]
"""
transform = transforms.Compose([
    transforms.ToTensor()
])

# ==================Dataset==================
"""
torch.utils.data.Dataset
功能：Dataset 是抽象类，所有自定义的 Dataset 都需要继承该类，并且重写__getitem()__方法和__len__()方法 。
__getitem()__方法的作用是接收一个索引，返回索引对应的样本和标签，这是我们自己需要实现的逻辑。__len__()方法是返回所有样本的数量。

数据读取包含 3 个方面:
读取哪些数据：每个 Iteration 读取一个 Batch_size 大小的数据，每个 Iteration 应该读取哪些数据。
从哪里读取数据：如何找到硬盘中的数据，应该在哪里设置文件路径参数
如何读取数据：不同的文件需要使用不同的读取方法和库。
"""

# ================================Dataset==================
"""
class Dataset(typing.Generic)
 |  An abstract class representing a :class:`Dataset`.
 |  
 |  All datasets that represent a map from keys to data samples should subclass
 |  it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
 |  data sample for a given key. Subclasses could also optionally overwrite
 |  :meth:`__len__`, which is expected to return the size of the dataset by many
 |  :class:`~torch.utils.data.Sampler` implementations and the default options
 |  of :class:`~torch.utils.data.DataLoader`.
 |  
 |  .. note::
 |    :class:`~torch.utils.data.DataLoader` by default constructs a index
 |    sampler that yields integral indices.  To make it work with a map-style
 |    dataset with non-integral indices/keys, a custom sampler must be provided.
 |  
 |  Method resolution order:
 |      Dataset
 |      typing.Generic
 |      builtins.object
 |  
 |  Methods defined here:
 |  
 |  __add__(self, other: 'Dataset[T_co]') -> 'ConcatDataset[T_co]'
 |  
 |  __getitem__(self, index) -> +T_co
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |  
 |  __dict__
 |      dictionary for instance variables (if defined)
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)
 |  
 |  ----------------------------------------------------------------------
 |  Data and other attributes defined here:
 |  
 |  __orig_bases__ = (typing.Generic[+T_co],)
 |  
 |  __parameters__ = (+T_co,)
 |  
 |  ----------------------------------------------------------------------
 |  Class methods inherited from typing.Generic:
 |  
 |  __class_getitem__(params) from builtins.type
 |  
 |  __init_subclass__(*args, **kwargs) from builtins.type
 |      This method is called when a class is subclassed.
 |      
 |      The default implementation does nothing. It may be
 |      overridden to extend subclasses.

"""
# 定义数据集
class MyDataset(Dataset):
    def __init__(self, path):  # 定义路径
        self.path = path
        self.name = os.listdir(os.path.join(path, 'HO_image'))

    def __len__(self):  # 获取样本数量
        return len(self.name)

    def __getitem__(self, index):  # 接收索引，返回对应的数据和标签
        segment_name = self.name[index]  # xx.png
        segment_path = os.path.join(self.path, 'HO_image', segment_name)  # 根据数据集图片名称进行对应
        image_path = os.path.join(self.path, 'crop_image', segment_name)
        segment_image = keep_image_size_open(segment_path)
        image = keep_image_size_open(image_path)
        return transform(image), transform(segment_image)

# ====================Dataloader====================
"""
Dataloader
功能：构建可迭代的数据装载器
dataset: Dataset 类，决定数据从哪里读取以及如何读取
batch_size: 批大小
num_works:num_works: 是否多进程读取数据
shuffle: 每个 epoch 是否乱序
drop_last: 当样本数不能被 batch_size 整除时，是否舍弃最后一批数据
Epoch, Iteration, Batch_size
Epoch: 所有训练样本都已经输入到模型中，称为一个 Epoch
Iteration: 一批样本输入到模型中，称为一个 Iteration
Batch_size: 批大小，决定一个 iteration 有多少样本，也决定了一个 Epoch 有多少个 Iteration

"""

torch.utils.data.DataLoader(MyDataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0,
                            collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None,
                            multiprocessing_context=None)

# 定义训练
"""
开始训练，循环epochs(包括前项传播和反向传播)
---将梯度清零
---求loss
---反向传播
---更新权重参数
---更新优化器中的学习率(可选，优化器一般含有学习率)
"""

# =============================================深度学习模板===================================================================
"""
深度学习模板
1、导入包以及设置随机种子
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
 
import random
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


2、以类的方式定义超参数
class argparse():
    pass
 
args = argparse()
args.epochs, args.learning_rate, args.patience = [30, 0.001, 4]
args.hidden_size, args.input_size= [40, 30]
args.device, = [torch.device("cuda:0"if torch.cuda.is_available() else"cpu"),]


3、定义自己的模型
class Your_model(nn.Module):
    def __init__(self):
        super(Your_model, self).__init__()
        pass
        
    def forward(self,x):
        pass
        return x
        
        
        
4、定义自己的数据集Dataset，Dataloader
class Dataset_name(Dataset):
    def __init__(self, flag='train'):
        assert flag in ['train', 'DATA', 'valid']
        self.flag = flag
        self.__load_data__()
 
    def __getitem__(self, index):
        pass
    def __len__(self):
        pass
 
    def __load_data__(self, csv_paths: list):
        pass
        print(
            "train_X.shape:{}\ntrain_Y.shape:{}\nvalid_X.shape:{}\nvalid_Y.shape:{}\n"
            .format(self.train_X.shape, self.train_Y.shape, self.valid_X.shape, self.valid_Y.shape))
 
train_dataset = Dataset_name(flag='train')
train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
valid_dataset = Dataset_name(flag='valid')
valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=64, shuffle=True)


5、实例化模型，设置loss，优化器
model = Your_model().to(args.device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(Your_model.parameters(),lr=args.learning_rate)
 
train_loss = []
valid_loss = []
train_epochs_loss = []
valid_epochs_loss = []
 
early_stopping = EarlyStopping(patience=args.patience,verbose=True)


6、开始训练以及调整lr
for epoch in range(args.epochs):
    Your_model.train()
    train_epoch_loss = []
    for idx,(data_x,data_y) in enumerate(train_dataloader,0):
        data_x = data_x.to(torch.float32).to(args.device)
        data_y = data_y.to(torch.float32).to(args.device)
        outputs = Your_model(data_x)
        optimizer.zero_grad()
        loss = criterion(data_y,outputs)
        loss.backward()
        optimizer.step()
        train_epoch_loss.append(loss.item())
        train_loss.append(loss.item())
        if idx%(len(train_dataloader)//2)==0:
            print("epoch={}/{},{}/{}of train, loss={}".format(
                epoch, args.epochs, idx, len(train_dataloader),loss.item()))
    train_epochs_loss.append(np.average(train_epoch_loss))
    
    #=====================valid============================
    Your_model.eval()
    valid_epoch_loss = []
    for idx,(data_x,data_y) in enumerate(valid_dataloader,0):
        data_x = data_x.to(torch.float32).to(args.device)
        data_y = data_y.to(torch.float32).to(args.device)
        outputs = Your_model(data_x)
        loss = criterion(outputs,data_y)
        valid_epoch_loss.append(loss.item())
        valid_loss.append(loss.item())
    valid_epochs_loss.append(np.average(valid_epoch_loss))
    #==================early stopping======================
    early_stopping(valid_epochs_loss[-1],model=Your_model,path=r'c:\\your_model_to_save')
    if early_stopping.early_stop:
        print("Early stopping")
        break
    #====================adjust lr========================
    lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
        
        
        
7、绘图
plt.figure(figsize=(12,4))
plt.subplot(121)
plt.plot(train_loss[:])
plt.title("train_loss")
plt.subplot(122)
plt.plot(train_epochs_loss[1:],'-o',label="train_loss")
plt.plot(valid_epochs_loss[1:],'-o',label="valid_loss")
plt.title("epochs_loss")
plt.legend()
plt.show()

8、预测
# 此处可定义一个预测集的Dataloader。也可以直接将你的预测数据reshape,添加batch_size=1
Your_model.eval()
predict = Your_model(data)
"""

# ================================================================================================
if __name__ == '__main__':
    data = MyDataset(r'D:\Project image\particle')
    print(data[0][0].shape)
    print(data[0][1].shape)
