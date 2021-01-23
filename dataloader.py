#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import random
from PIL import Image

class YouTubePose():
    def __init__(self, dataset_dir, class_num, transform = None):
        self.dataset_dir = dataset_dir
        self.class_num = class_num
        self.transform = transform
          
    def __len__(self):
        length = []
        for i in range(self.class_num):
            dir_path = self.dataset_dir +'train/class{}_cropped'.format(i + 1)
            length.append(len(os.walk(dir_path).__next__()[2]))
        max_len = max(length)
        return max_len    
    
    def __getitem__(self, idx):
        randx, randy = random.sample(range(1, self.class_num + 1), 2)
        x, x_hat, identity_equal_1, identity_equal_2 = random.sample(os.listdir(self.dataset_dir + 
                                 'train/class{}_cropped'.format(randx)), 4)
        y = random.choice(os.listdir(self.dataset_dir + 
                                 'train/class{}_cropped'.format(randy)))
          
        x = Image.open(self.dataset_dir + 'train/class{}_cropped/'
                       .format(randx)+ x)
        y = Image.open(self.dataset_dir + 'train/class{}_cropped/'
                       .format(randy)+ y)
        x_hat = Image.open(self.dataset_dir + 'train/class{}_cropped/'
                           .format(randx)+ x_hat)
        
        identity_equal_1 = Image.open(self.dataset_dir + 'train/class{}_cropped/'
                                      .format(randx)+ identity_equal_1)
        identity_equal_2 = Image.open(self.dataset_dir + 'train/class{}_cropped/'
                                      .format(randx)+ identity_equal_2)
        
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
            x_hat = self.transform(x_hat)
            identity_equal_1 = self.transform(identity_equal_1)
            identity_equal_2 = self.transform(identity_equal_2)
       
        sample = {'x' : x, 'y' : y, 'x_hat' : x_hat, 
                 'identity_equal_1' : identity_equal_1, 
                 'identity_equal_2' : identity_equal_2}
        
        return sample

