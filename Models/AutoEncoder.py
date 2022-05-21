#Import Libraries 

import torch 
import torch.nn as nn

#TODO: Define the Stride and Kernel size based on the size of Images
#TODO: Define the Size of the images
#SUGGESTED : Define the linear layers for better training embedding feature
#TODO : Define the epochs and batch sizes to be used for incremental training

class Net(nn.Module):
    def __init__(self , Encoder, Decoder):
        super(Net, self).__init__()
        self.Encoder_layer = Encoder
        self.Decoder_layer = Decoder


    def forward(self, x):
        x = self.Encoder_layer(x)
        x = self.Decoder_layer(x)
        return x


class Encoder(nn.Module):
    def __init__(self):

        super(Encoder, self).__init__()
        self.enc1 = nn.Conv3d(64, 64)
        
        self.enc2 = nn.Conv3d(64, 32)
        self.enc3 = nn.Conv3d(32, 32)
        
        self.enc4 = nn.Conv3d(32, 16)
        self.enc5 = nn.Conv3d(16, 16)
        
        self.enc6 = nn.Conv3d(16, 8)
        self.enc7 = nn.Conv3d(8, 8)
        
        self.enc8 = nn.Conv3d(8, 4) 
        self.enc9 = nn.Conv3d(4, 4)
        
        self.enc10 = nn.Conv3d(4, 4) # TODO : Define kernel and stride
        
        
    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        x = self.enc6(x)
        x = self.enc7(x)
        x = self.enc8(x)
        x = self.enc9(x)
        x = self.enc10(x)
    

        return x

class Decoder(nn.Module):
    def __init__(self):

        super(Decoder, self).__init__()
        self.dec1 = nn.ConvTranspose3d(4, 4)
        self.dec2 = nn.Conv3d(4, 4)

        self.dec3 = nn.ConvTranspose3d(4, 8)
        self.dec4 = nn.Conv3d(8, 8)

        self.dec5 = nn.ConvTranspose3d(8, 16)
        self.dec6 = nn.Conv3d(16, 16)

        self.dec7 = nn.ConvTranspose3d(16, 32)
        self.dec8 = nn.Conv3d(32, 32)
        
        self.dec9 = nn.ConvTranspose3d(32, 64)
        self.dec10 = nn.Conv3d(64, 64)

    def forward(self, x):
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)
        x = self.dec6(x)
        x = self.dec7(x)
        x = self.dec8(x)
        x = self.dec9(x)
        x = self.dec10(x)
        
        return x