import torch
import torch.nn as nn
from blocks import *

input_channel = 3
channel = 64

class Generator(nn.Module):
    def __init__(self, resblock, padding_mode = 'zeros'):
        super(Generator, self).__init__()
        
        self.x_conv1 = ConvBlock(input_channel, channel, 7, 1, 3)
        self.y_conv1 = ConvBlock(input_channel, channel, 7, 1, 3)
        
        self.x_conv2 = ConvBlock(channel, channel * 2, 3, 2, 1)
        self.y_conv2 = ConvBlock(channel, channel * 2, 3, 2, 1)
        
        self.x_conv3 = ConvBlock(channel * 2, channel * 4, 3, 2, 1)        
        self.y_conv3 = ConvBlock(channel * 2, channel * 4, 3, 2, 1)
        
        self.x_resblock = resblock(channel * 4, channel * 4)
        self.y_resblock = resblock(channel * 4, channel * 4)
        
        self.resblock = resblock(channel * 4, channel * 4)
        
        self.deconv1 = ConvBlock(channel * 4, channel * 2, 4, 2, 1, upsampling = True)
        self.deconv2 = ConvBlock(channel * 2, channel, 4, 2, 1, upsampling = True)
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(channel, input_channel, 7, 1, 3, padding_mode = padding_mode),
            nn.Tanh(),
        )

    def forward(self, x, y):
        x_out = self.x_conv1(x) #size는 그대로 channel만 두배
        x_out = self.x_conv2(x_out) #size 1/2배 channel 두배
        x_out_128 = x_out
        x_out = self.x_conv3(x_out)
        x_out_64 = x_out
        x_out = self.x_resblock(x_out)
        x_out = self.x_resblock(x_out)
        
        y_out = self.y_conv1(y)
        y_out = self.y_conv2(y_out)
        y_out_128 = y_out
        y_out = self.y_conv3(y_out)
        y_out_64 = y_out
        y_out = self.y_resblock(y_out)
        y_out = self.y_resblock(y_out)
        
        out = torch.add(x_out, y_out)
        out = self.resblock(out)
        out = self.resblock(out)
        out = torch.add(out, x_out_64)
        out = torch.add(out, y_out_64)

        out = self.deconv1(out)
        out = torch.add(out, x_out_128)
        out = torch.add(out, y_out_128)
        out = self.deconv2(out)
        out = self.deconv3(out)
        return out

class Discriminator(nn.Module): 
    def __init__(self, padding_mode = 'zeros'):
        super(Discriminator, self).__init__() 
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel * 2, channel, 4, 2, 1, padding_mode = padding_mode),
            nn.LeakyReLU(0.2),
        )
        self.conv2 = ConvBlock(channel, channel * 2, 4, 2, 1, Leaky = True)
        self.conv3 = ConvBlock(channel * 2, channel * 4, 4, 2, 1, Leaky = True)
        self.conv4 = ConvBlock(channel * 4, channel * 8, 3, 1, 1, Leaky = True)
        self.conv5 = ConvBlock(channel * 8, channel * 8, 3, 1, 1, Leaky = True)
        self.conv6 = nn.Sequential(
            nn.Conv2d(channel * 8, 1, 3, 1, 1, padding_mode = padding_mode),
            nn.Sigmoid()
        )
    
    def forward(self,x, y):
        xy = torch.cat([x,y], dim = 1)
        out = self.conv1(xy)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        return out

