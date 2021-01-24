import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from models import *
from utils import *
from dataloader import *

rand_fix()

LI_G_list = []
LI_D_list = []
Ls1_list = []
Ls2_list = []

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
dataset_dir = './Dataset/'
transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
class_num = 3

num_workers = 0
batch_size = 4
num_epochs = 100
lr_D = 0.003
lr_G = 0.001
alpha = 0.8
beta = 0.3

save_dir = './CheckPoint/'

generator = Generator(ResidualBlock).to(device)
generator.apply(weights_init)

discriminator = Discriminator().to(device)
discriminator.apply(weights_init)

optimizer_G = optim.Adam(generator.parameters(), lr = lr_G, betas = (0.5,0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr = lr_D, betas = (0.5, 0.999))


real_label = 1.
fake_label = 0.

train_dataset = YouTubePose(dataset_dir, class_num, transform)
train_dataloader = DataLoader(train_dataset, batch_size = batch_size,
                             shuffle = True, num_workers = num_workers)

######### Start Training##########
Start = time.time()
dataloader = train_dataloader

netG = generator
netG.train()
netD = discriminator
netD.train()

iden_loss = nn.MSELoss()
shape_loss = nn.L1Loss()

for epoch in range(num_epochs):
    print("Epoch {}/{}".format(epoch + 1, num_epochs))
    print('-' * 10)
    start = time.time()
    for i, sample in enumerate(dataloader, 0): #start = 0
        x, y, x_hat = sample['x'], sample['y'], sample['x_hat']
        iden_equal1 = sample['identity_equal_1']
        iden_equal2 = sample['identity_equal_2']
        batch_size = x.size(0)
        #with torch.autograd.detect_anomaly(): 
        x = x.to(device)
        y = y.to(device)
        x_hat = x_hat.to(device)
        iden_equal1 = iden_equal1.to(device)
        iden_equal2 = iden_equal2.to(device)

        optimizer_G.zero_grad()
        optimizer_D.zero_grad()

        #Ix != Iy
        #Update G with maximize LI
        output = netG(x, y)
        fake= netD(x, output)
        real= netD(x, x_hat)
        Real_label = torch.full((real.size()), real_label, dtype=torch.float, device=device)
        Fake_label = torch.full((fake.size()), fake_label, dtype=torch.float, device=device)
        iden_loss_G = iden_loss(Real_label, real) + iden_loss(Fake_label, fake)
        LI_G = -iden_loss_G  #maximize LI
        LI_G_list.append(LI_G.item())
        LI_G.backward(retain_graph=True) #역전파 두번 사용시 필요
        optimizer_G.step()

        #Update D with minimize LI
        optimizer_D.zero_grad()
        Real_label = torch.full((real.size()), real_label, dtype=torch.float, device=device)
        Fake_label = torch.full((fake.size()), fake_label, dtype=torch.float, device=device)
        if(-LI_G <= 0.1):
            iden_loss_D = iden_loss(Fake_label, real) + iden_loss(Real_label, fake)
        else:
            iden_loss_D = iden_loss(Real_label, real) + iden_loss(Fake_label, fake)
        LI_D = iden_loss_D
        LI_D_list.append(LI_D.item())
        LI_D.backward(retain_graph=True)
        optimizer_D.step()

        #Update G with minimize Ls2a, Ls2b
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()
        output_ls2a = netG(y, output)
        output_ls2b = netG(output, y)

        Ls2a = shape_loss(y, output_ls2a)
        Ls2b = shape_loss(output, output_ls2b)
        Ls2 = (Ls2a + Ls2b) * alpha
        Ls2_list.append(Ls2.item())

        Ls2.backward()
        optimizer_G.step()

        #Ix == Iy
        #Update G with Ls1
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()

        output_ls1 = netG(iden_equal1, iden_equal2)

        Ls1 = shape_loss(iden_equal2, output_ls1) * beta
        Ls1_list.append(Ls1.item()) 

        Ls1.backward()
        optimizer_G.step()

        if (i % 1000 == 0):
            print("[{:d}/{:d}] LI_G:{:.4f}    LI_D:{:.4f}    Ls1:{:.4f}     Ls2a:{:.4f}     Ls2b:{:.4f}".
         format(i, len(dataloader), LI_G, LI_D, Ls1, Ls2a, Ls2b))

    save_checkpoint({
                'epoch': epoch + 1,
                'netG_state_dict': netG.state_dict(),
                'netD_state_dict': netD.state_dict(),
                'gen_opt': optimizer_G.state_dict(),
                'disc_opt': optimizer_D.state_dict()
            }, save_dir, epoch + 1)    

    print("="*100)
    print('Time taken by 1epoch: {:.0f}h {:.0f}m {:.0f}s'.format(((time.time() - start) // 60) // 60, ((time.time() - start) // 60) % 60, (time.time() - start) % 60))
    print()

    x = x.detach().cpu()
    y = y.detach().cpu()
    result = output.detach().cpu()
    sample = []
    for i in range(batch_size):
        sample.extend([x[i], y[i], result[i]])
    result_img = utils.make_grid(sample, padding = 2,
                                   normalize = True, nrow = 3)
    utils.save_image(result_img, "./result/{}epoch.png".format(epoch + 1))

print("Training is finished")
print('Time taken by {}epochs: {:.0f}h {:.0f}m {:.0f}s'.format(num_epochs, ((time.time() - Start) // 60) // 60, ((time.time() - Start) // 60) % 60, (time.time() - Start) % 60))

