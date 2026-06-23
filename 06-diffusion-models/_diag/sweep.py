import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader, Subset

torch.manual_seed(0)
root = Path('C:/Users/tivog/deep-generative-models')
dev = torch.device('cpu')
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
    def reparameterize(self,mu,lv):
        return mu + torch.exp(0.5*lv)*torch.randn_like(mu)
    def decode(self,z): return self.decoder(z)
    def forward(self,x):
        mu,lv=self.encode(x); z=self.reparameterize(mu,lv); return self.decode(z),mu,lv

tf=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x:2*x-1)])
full=datasets.FashionMNIST(root=root/'data', train=True, download=True, transform=tf)
ds=Subset(full, range(20000))
dl=DataLoader(ds,batch_size=128,shuffle=True)
xb,_=next(iter(DataLoader(full,batch_size=8,shuffle=True)))

for beta in [0.002, 0.01, 0.05]:
    torch.manual_seed(0)
    vae=ConvVAE(lc).to(dev)
    opt=torch.optim.Adam(vae.parameters(),lr=2e-4)
    for ep in range(8):
        for x,_ in dl:
            recon,mu,lv=vae(x)
            recon_loss=F.l1_loss(recon,x)+F.mse_loss(recon,x)
            kl=-0.5*torch.mean(1+lv-mu.pow(2)-lv.exp())
            loss=recon_loss+beta*kl
            opt.zero_grad(); loss.backward(); opt.step()
    vae.eval()
    with torch.no_grad():
        mu,lv=vae.encode(xb)
        s=mu.std().item(); pstd=(0.5*lv).exp().mean().item()
        recon=vae.decode(mu).clamp(-1,1)
        gaus=vae.decode(s*torch.randn_like(mu)).clamp(-1,1)
        grid=utils.make_grid(torch.cat([0.5*(xb+1),0.5*(recon+1),0.5*(gaus+1)]),nrow=8,pad_value=1.0)
        utils.save_image(grid, root/'06-diffusion-models'/'_diag'/f'sweep_b{beta}.png')
    print(f'beta={beta}: mu_std={s:.3f} post_std={pstd:.3f} recon={recon_loss.item():.4f} -> sweep_b{beta}.png (rows: orig/recon/gaussian-decode)')
