class residual_block(nn.Module):
    def __init__(self):
        super(residual_block,self).__init__()
        self.conv1 = nn.Conv2d(64,64, kernel_size=3, stride=1,padding= 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.LeakyReLU(negative_slope=0.02, inplace=True)
        self.conv2 = nn.Conv2d(64,64, kernel_size=3, stride=1,padding= 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.LeakyReLU(negative_slope=0.02, inplace=True)

    def forward(self,x):
        x_p=x
        x= self.conv1(x)     
        x= self.bn1(x)
        x=self.relu1(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x= x_p + x 
        x=self.relu(x)
        return x


class colorization(nn.Module):
    def __init__(self, block, layers, channels, num_classes):
        
        super(colorization, self).__init__()
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
    
        self.residual_block= residual_block() 

        self.conv2 = nn.Conv2d(64, 3, kernel_size= 1, stride=1, padding=0, bias=False)
        self.deconv= nn.ConvTranspose2d(3, 3, 8, stride=4, padding=2, group=3, bias=False)
    
    def forward(self,x):

        x=self.conv1(x)
        
        x=self.bn1(x) 
        x=self.relu(x) 
        x=self.maxpool(x)

        for i in range(7):
            x=self.residual_block(x)

        x=self.conv2(x) 
        x=self.deconv(x)
        return x
