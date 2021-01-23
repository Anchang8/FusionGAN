import torch
import torch.nn as nn

def _conv(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1,
            upsampling = False, padding_mode = 'zeros'):
    if(upsampling):
        conv = nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size = kernel_size, 
                                                          stride = stride, padding = padding, padding_mode = padding_mode))
    else:
        conv = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size,
                    stride = stride, padding = padding, padding_mode = padding_mode))
    return conv

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = _conv(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = _conv(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        residual = x
        if torch.cuda.is_available():
             x = torch.cuda.FloatTensor(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.relu(out) 
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 upsampling = False, padding_mode = 'zeros', Leaky = False):
        super(ConvBlock, self).__init__()
        self.conv = _conv(in_channels, out_channels, kernel_size, stride, padding, upsampling, padding_mode)
        self.bn = nn.BatchNorm2d(out_channels)
        if (Leaky):
            self.relu = nn.LeakyReLU(0.2)
        else:
            self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out