import torch, torch.nn as nn
from pathlib import Path
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader

root = Path('C:/Users/tivog/deep-generative-models')
lc = 8

class ConvVAE(nn.Module):
    def __init__(self, latent_channels=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,32,4,2,1), nn.SiLU(),
            nn.Conv2d(32,64,4,2,1), nn.SiLU(),
            nn.Conv2d(64,128,3,padding=1), nn.SiLU())
        self.mu_head = nn.Conv2d(128, latent_channels, 3, padding=1)
        self.logvar_head = nn.Conv2d(128, latent_channels, 3, padding=1)
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels,128,3,padding=1), nn.SiLU(),
            nn.ConvTranspose2d(128,64,4,2,1), nn.SiLU(),
            nn.ConvTranspose2d(64,32,4,2,1), nn.SiLU(),
            nn.Conv2d(32,1,3,padding=1), nn.Tanh())
    def encode(self,x):
        h=self.encoder(x); return self.mu_head(h), self.logvar_head(h)
    def decode(self,z): return self.decoder(z)

vae=ConvVAE(lc); vae.load_state_dict(torch.load(root/'artifacts'/'fashionmnist_ldm_vae.pt',map_location='cpu')); vae.eval()
tf=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x:2*x-1)])
ds=datasets.FashionMNIST(root=root/'data', train=True, download=True, transform=tf)
x,_=next(iter(DataLoader(ds,batch_size=8,shuffle=True)))
with torch.no_grad():
    mu,_=vae.encode(x)
    rows=[0.5*(x+1)]
    for s in [0.1,0.3,0.5,1.0]:
        dec=vae.decode(mu+s*torch.randn_like(mu)).clamp(-1,1)
        rows.append(0.5*(dec+1))
    # also decode a pure-Gaussian latent at the correct scale (std 2.13)
    rand=vae.decode(2.13*torch.randn_like(mu)).clamp(-1,1)
    rows.append(0.5*(rand+1))
grid=utils.make_grid(torch.cat(rows),nrow=8,pad_value=1.0)
utils.save_image(grid, root/'06-diffusion-models'/'_diag'/'perturb.png')
print('rows: original, +N(0,.1), +.3, +.5, +1.0, pure-gaussian@2.13')
print('saved perturb.png')
