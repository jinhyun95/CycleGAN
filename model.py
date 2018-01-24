import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# single generator and single discriminator class is needed for image2image task
# for tasks with different input and output dataform, two different generator modules need to be defined
# following parameters are based on official Github repo of Cycle-GAN or the paper itself
class Resnet(nn.Module):
    def __init__(self, dim):
        super(Resnet, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=dim)
        )
    def forward(self, x):
        return x + self.residual(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.network = nn.Sequential(
            # 3*256*256 --> 32*256*256
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            # 32*256*256 --> 64*128*128
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            # 64*128*128 --> 128*64*64
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            Resnet(dim=128),
            Resnet(dim=128),
            Resnet(dim=128),
            Resnet(dim=128),
            Resnet(dim=128),
            Resnet(dim=128),
            Resnet(dim=128),
            Resnet(dim=128),
            Resnet(dim=128),
            # 128*64*64 --> 64*128*128
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            # 64*128*128 --> 32*256*256
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            # 32*256*256 --> 3*256*256
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=7, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

        # Disco-GAN generator structure
        # self.network = nn.Sequential(
        #     # convolution
        #     # 3*256*256 --> 64*128*128
        #     nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     # 64*128*128 --> 128*64*64
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(num_features=128),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     # 128*64*64 --> 256*32*32
        #     nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(num_features=256),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     # 256*32*32 --> 512*16*16
        #     nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(num_features=512),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     # 512*16*16 --> 1024*8*8
        #     nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(num_features=1024),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     # deconvolution
        #     # 1024*8*8 --> 512*16*16
        #     nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(num_features=512),
        #     nn.ReLU(inplace=True),
        #     # 512*16*16 --> 256*32*32
        #     nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(num_features=256),
        #     nn.ReLU(inplace=True),
        #     # 256*32*32 --> 128*64*64
        #     nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(num_features=128),
        #     nn.ReLU(inplace=True),
        #     # 128*64*64 --> 64*128*128
        #     nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(num_features=64),
        #     nn.ReLU(inplace=True),
        #     # 64*128*128 --> 3*256*256
        #     nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
        #     nn.Sigmoid()
        # )

    def forward(self, image):
        return self.network(image)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
            # 3*256*256 --> 64*128*128
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # 64*128*128 --> 128*64*64
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # 128*64*64 --> 256*32*32
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # 256*32*32 --> 512*16*16
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # 512*16*16 --> 1*1*1
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=16, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, image):
        return self.network(image)