import torch, torch.nn as nn
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device('cpu')
root = Path('C:/Users/tivog/deep-generative-models')
latent_channels = 8

class ConvVAE(nn.Module):
    def __init__(self, latent_channels=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,32,4,2,1), nn.SiLU(),
            nn.Conv2d(32,64,4,2,1), nn.SiLU(),
            nn.Conv2d(64,128,3,padding=1), nn.SiLU())
        self.mu_head = nn.Conv2d(128, latent_channels, 3, padding=1)
        self.logvar_head = nn.Conv2d(128, latent_channels, 3, padding=1)
    def encode(self,x):
        h=self.encoder(x); return self.mu_head(h), self.logvar_head(h)

vae=ConvVAE(latent_channels).to(device)
sd=torch.load(root/'artifacts'/'fashionmnist_ldm_vae.pt', map_location='cpu')
vae.load_state_dict(sd, strict=False)
vae.eval()

tf=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x:2*x-1)])
ds=datasets.FashionMNIST(root=root/'data', train=True, download=True, transform=tf)
dl=DataLoader(ds,batch_size=256,shuffle=False)

mus=[]; lvs=[]
with torch.no_grad():
    for i,(x,_) in enumerate(dl):
        mu,lv=vae.encode(x.to(device)); mus.append(mu); lvs.append(lv)
        if i>=15: break
mu=torch.cat(mus); lv=torch.cat(lvs)
print('mu shape', mu.shape)
print('mu  global std =', mu.std().item(), ' mean =', mu.mean().item())
print('mu  min/max =', mu.min().item(), mu.max().item())
print('per-channel std:', [round(mu[:,c].std().item(),3) for c in range(latent_channels)])
print('logvar mean =', lv.mean().item(), ' -> avg posterior std =', (0.5*lv).exp().mean().item())
print('what 0.18215 scaling does: latents = mu/0.18215 -> std =', (mu/0.18215).std().item())
print('correct scaling would divide by', round(mu.std().item(),4), '-> std ~1')
