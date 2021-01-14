import torch.nn as nn
# import torch

model_architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,**kwargs):
        super(CNNBlock,self).__init__()
        self.block = nn.Sequential(
                        nn.Conv2d(in_channels,out_channels,kernel_size,bias=False,**kwargs),
                        nn.BatchNorm2d(out_channels),
                        nn.LeakyReLU(0.1)
                        )
    
    def forward(self,x):
        return self.block(x)
        
    
    
class YOLOv1(nn.Module):
    def __init__(self,in_channels=3,**kwargs):
        super(YOLOv1,self).__init__()
        self.architecture = model_architecture_config
        self.in_channels = in_channels
        self.darknet = self.create_darknet(self.architecture)
        self.fcl = self.create_fcl(**kwargs)
        
    def forward(self,x):
        for module in self.darknet:
            x = module(x)
        for module in self.fcl:
            x = module(x)
        
        return x

    def create_darknet(self,architecture):
        modules = nn.ModuleList()
        in_channels = self.in_channels
        
        for module in architecture:
            if type(module)==tuple:
                modules.append(CNNBlock(in_channels,module[1],module[0],stride=module[2],padding=module[3]))        
                in_channels = module[1]
                
            elif type(module)==str:
                modules.append(nn.MaxPool2d(kernel_size=2 ,stride=2))
                
            elif type(module)==list:
                num_repeats = module[2]
                conv1 = module[0]
                conv2 = module[1]
                
                for i in range(num_repeats):
                    modules.append(CNNBlock(in_channels,conv1[1],conv1[0],stride=conv1[2],padding=conv1[3]))
                    in_channels = conv1[1]
                    modules.append(CNNBlock(in_channels,conv2[1],conv2[0],stride=conv2[2],padding=conv2[3]))
                    in_channels = conv2[1]
                    
        return modules
    
    def create_fcl(self,grid_size,num_bbox,num_classes):
        layer = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(1024*grid_size*grid_size,496),
                    nn.Dropout(0.0),
                    nn.LeakyReLU(0.1),
                    nn.Linear(496,grid_size*grid_size*(num_bbox*5+num_classes))
                    )
        modules = nn.ModuleList()
        modules.append(layer)
        return modules
                
# model = YOLOv1(grid_size=7,num_bbox=2,num_classes=20)  
# # print(model)
# x = torch.rand((2,3,448,448))      
# print(model(x).shape)
        
        
        
        
        
        
        