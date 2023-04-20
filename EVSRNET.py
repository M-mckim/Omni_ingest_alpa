import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1)
        
    def forward(self, x):
        fx = self.conv1(x)
        fx = self.relu(fx)
        fx = self.conv2(fx)
        out = x + fx
        out = self.relu(out)
        
        return out
    
class EVSRNet(nn.Module):
    def __init__(self, scale_factor=4, input_shape=(3, None, None)):
        super(EVSRNet, self).__init__()
        self.scale_factor = scale_factor
        self.input_shape = input_shape
        
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=8, kernel_size=3, padding=1)
        self.res1 = ResidualBlock()
        self.res2 = ResidualBlock()
        self.res3 = ResidualBlock()
        self.res4 = ResidualBlock()
        self.res5 = ResidualBlock()
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=input_shape[0] * (scale_factor ** 2), kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.depth_to_space = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.conv2(x)
        x = self.relu(x)
        out = self.depth_to_space(x)
        
        return out