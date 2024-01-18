import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

class UnetBlock(nn.Module):
  def __init__(self, in_chan, out_chan, batch_norm, device):
    super(UnetBlock, self).__init__()
    self.in_chan = in_chan
    self.out_chan = out_chan
    self.batch_norm = batch_norm
    self.device = device
    self.conv1 = nn.Conv2d(in_channels=self.in_chan,
                  out_channels=self.out_chan,
                  padding="same",
                  kernel_size=3,
                  device=self.device)
    self.conv2 = nn.Conv2d(in_channels=self.out_chan, 
                  out_channels=self.out_chan, 
                  kernel_size=(3,3), 
                  padding="same",
                  device=self.device)
  
  def forward(self, x):
    x = self.conv1(x)
    if self.batch_norm:
      x = nn.BatchNorm2d(self.out_chan, device=self.device)(x)

    x = F.leaky_relu(x)
    x = self.conv2(x)
    if self.batch_norm:
      x = nn.BatchNorm2d(self.out_chan, device=self.device)(x)
    return F.leaky_relu(x)

  
class Encoder(nn.Module):
  def __init__(self, 
               nb_levels,
               nb_filters,
               batch_norm, 
               chan_in,
               device):
    
    super(Encoder, self).__init__()
    self.nb_levels = nb_levels
    self.nb_filters = nb_filters
    self.batch_norm = batch_norm
    self.chan_in = chan_in
    self.blocks = [UnetBlock(chan_in, self.nb_filters, self.batch_norm, device)]
    self.downs = []
    # self.params = []
    self.device = device

    for lev in range(1, self.nb_levels - 1):
      self.downs.append(nn.Conv2d(lev*self.nb_filters, (lev+1)*self.nb_filters, kernel_size=2, stride=2, device=self.device))
      self.blocks.append(UnetBlock((lev+1)*self.nb_filters, (lev+1)*self.nb_filters, self.batch_norm, self.device))

    self.downs.append(nn.Conv2d((self.nb_levels - 1)*self.nb_filters,
                                self.nb_levels*self.nb_filters,
                                kernel_size=2, stride=2, device=self.device))

    self.blocks = nn.ModuleList(self.blocks)
    self.downs = nn.ModuleList(self.downs)

  def forward(self, x):
    skip_co = []
    for b, d in zip(self.blocks, self.downs):
      x = b(x)
      skip_co.append(x)
      x = d(x)
    return x, skip_co


class Decoder(nn.Module):
  def __init__(self,
               nb_levels,
               nb_filters,
               batch_norm,
               device):
    super(Decoder, self).__init__()
    self.nb_levels = nb_levels
    self.nb_filters = nb_filters
    self.batch_norm = batch_norm
    self.device = device

    self.blocks = []
    self.up = []

    for lev in range(1, self.nb_levels):
      self.up.append(nn.ConvTranspose2d(in_channels=(self.nb_levels - lev + 1)*self.nb_filters,
                        out_channels=(self.nb_levels - lev)*self.nb_filters,
                        kernel_size=2,
                        stride=2,
                        device=self.device))
      
      self.blocks.append(UnetBlock(in_chan=2*(self.nb_levels - lev)*self.nb_filters,
                    out_chan=(self.nb_levels - lev)*self.nb_filters, 
                    batch_norm=self.batch_norm,
                    device=self.device))
    
    self.blocks = nn.ModuleList(self.blocks)
    self.up = nn.ModuleList(self.up)

  def forward(self, x, skip_co):
    for b, u in zip(self.blocks, self.up):
      x = u(x)
      skipped = skip_co.pop()
      x = torch.cat([skipped, x], dim=1)
      x = b(x)
    return x
