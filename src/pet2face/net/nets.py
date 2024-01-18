from .unet_blocks import *

class UNET(nn.Module):
  def __init__(self,
              device,
              nb_levels=4,
              nb_filters=32,
              input_channels=1,
              output_channels=1,
              sigma_noise=None,
              batch_norm=True,
              ):
    super(UNET, self).__init__()
    self.nb_levels = nb_levels
    self.nb_filters = nb_filters
    self.output_channels = output_channels
    self.noise = sigma_noise
    self.device = device
    
    self.encoder = Encoder(nb_levels, nb_filters, batch_norm, input_channels, device)
    self.decoder = Decoder(nb_levels, nb_filters, batch_norm, device)
    self.conv = nn.Conv2d(self.nb_filters, self.output_channels, kernel_size=1, device=self.device)

  def forward(self, x):
    x, skip_co = self.encoder(x)
    x = self.decoder(x, skip_co)
    if self.noise:
      x = x + self.noise*torch.randn_like(x)
    x = self.conv(x)
    x = torch.nn.Sigmoid()(x)
    return x