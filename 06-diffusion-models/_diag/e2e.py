import torch, torch.nn as nn, torch.nn.functional as F, math
from pathlib import Path
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader, Subset

torch.manual_seed(0)
root = Path('C:/Users/tivog/deep-generative-models')
dev = torch.device('cpu')
lc = 8; T = 300

# schedule (same as notebook)
betas = torch.linspace(1e-4, 0.02, T)
ap = 1.0 - betas
alphas = torch.cumprod(ap, 0)
sa = torch.sqrt(alphas); soma = torch.sqrt(1 - alphas)
srba = torch.sqrt(1.0 / ap)
ap_prev = torch.cat([torch.tensor([1.0]), alphas[:-1]])
post_var = betas * (1 - ap_prev) / (1 - alphas)
def ext(c, t, shp): return c.gather(0, t).view(t.shape[0], *([1]*(len(shp)-1)))
def q_sample(x0, t, n): return ext(sa,t,x0.shape)*x0 + ext(soma,t,x0.shape)*n

class ConvVAE(nn.Module):
    def __init__(s):
        super().__init__()
        s.encoder=nn.Sequential(nn.Conv2d(1,32,4,2,1),nn.SiLU(),nn.Conv2d(32,64,4,2,1),nn.SiLU(),nn.Conv2d(64,128,3,padding=1),nn.SiLU())
        s.mu_head=nn.Conv2d(128,lc,3,padding=1); s.lv_head=nn.Conv2d(128,lc,3,padding=1)
        s.decoder=nn.Sequential(nn.Conv2d(lc,128,3,padding=1),nn.SiLU(),nn.ConvTranspose2d(128,64,4,2,1),nn.SiLU(),nn.ConvTranspose2d(64,32,4,2,1),nn.SiLU(),nn.Conv2d(32,1,3,padding=1),nn.Tanh())
    def encode(s,x): h=s.encoder(x); return s.mu_head(h),s.lv_head(h)
    def decode(s,z): return s.decoder(z)
    def forward(s,x):
        mu,lv=s.encode(x); z=mu+torch.exp(0.5*lv)*torch.randn_like(mu); return s.decode(z),mu,lv

class SinEmb(nn.Module):
    def __init__(s,d): super().__init__(); s.d=d
    def forward(s,t):
        h=s.d//2; f=math.log(10000)/max(h-1,1)
        fr=torch.exp(torch.arange(h,device=t.device)*-f); a=t.float().unsqueeze(1)*fr.unsqueeze(0)
        return torch.cat([a.sin(),a.cos()],1)
class RB(nn.Module):
    def __init__(s,ci,co,cd):
        super().__init__(); s.n1=nn.GroupNorm(8,ci); s.c1=nn.Conv2d(ci,co,3,padding=1)
        s.n2=nn.GroupNorm(8,co); s.c2=nn.Conv2d(co,co,3,padding=1); s.cp=nn.Linear(cd,co)
        s.sk=nn.Conv2d(ci,co,1) if ci!=co else nn.Identity()
    def forward(s,x,c):
        h=s.c1(F.silu(s.n1(x))); h=h+s.cp(c)[:,:,None,None]; h=s.c2(F.silu(s.n2(h))); return h+s.sk(x)
class UNet(nn.Module):
    def __init__(s,td=128,bc=96):
        super().__init__()
        s.tm=nn.Sequential(SinEmb(td),nn.Linear(td,td),nn.SiLU(),nn.Linear(td,td))
        s.inp=nn.Conv2d(lc,bc,3,padding=1); s.d1=RB(bc,bc,td); s.ds=nn.Conv2d(bc,bc*2,4,2,1)
        s.m1=RB(bc*2,bc*2,td); s.m2=RB(bc*2,bc*2,td)
        s.us=nn.ConvTranspose2d(bc*2,bc,4,2,1,output_padding=1); s.u1=RB(bc*2,bc,td); s.out=nn.Conv2d(bc,lc,3,padding=1)
    def forward(s,z,t):
        c=s.tm(t); z0=s.inp(z); z1=s.d1(z0,c); z2=s.ds(z1); z2=s.m1(z2,c); z2=s.m2(z2,c)
        z3=s.us(z2); z3=torch.cat([z3,z1],1); z3=s.u1(z3,c); return s.out(z3)

tf=transforms.Compose([transforms.ToTensor(),transforms.Lambda(lambda x:2*x-1)])
full=datasets.FashionMNIST(root=root/'data',train=True,download=True,transform=tf)
ds=Subset(full,range(20000)); dl=DataLoader(ds,batch_size=128,shuffle=True)

# 1) train VAE with beta=0.01
vae=ConvVAE(); opt=torch.optim.Adam(vae.parameters(),2e-4)
for ep in range(12):
    for x,_ in dl:
        r,mu,lv=vae(x); rl=F.l1_loss(r,x)+F.mse_loss(r,x); kl=-0.5*torch.mean(1+lv-mu.pow(2)-lv.exp())
        loss=rl+0.01*kl; opt.zero_grad(); loss.backward(); opt.step()
print(f'VAE done recon={rl.item():.4f}')
vae.eval()
with torch.no_grad():
    coll=[vae.encode(x)[0].flatten() for i,(x,_) in enumerate(dl) if i<15]
    scaling=torch.cat(coll).std().item()
print(f'latent scaling={scaling:.4f}')

# 2) train unconditional latent diffusion
net=UNet(); opt=torch.optim.Adam(net.parameters(),2e-4)
for ep in range(25):
    tot=0;nb=0
    for x,_ in dl:
        with torch.no_grad(): z0=vae.encode(x)[0]/scaling
        t=torch.randint(0,T,(x.size(0),)); n=torch.randn_like(z0); zt=q_sample(z0,t,n)
        loss=F.mse_loss(net(zt,t),n); opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item(); nb+=1
    if ep%5==0 or ep==24: print(f'diff ep{ep} loss={tot/nb:.4f}')

# 3) sample
net.eval()
with torch.no_grad():
    z=torch.randn(16,lc,7,7)
    for ts in reversed(range(T)):
        t=torch.full((16,),ts,dtype=torch.long); eps=net(z,t)
        bt=ext(betas,t,z.shape); somat=ext(soma,t,z.shape); srbat=ext(srba,t,z.shape)
        mean=srbat*(z-bt*eps/somat)
        z=mean+(torch.sqrt(ext(post_var,t,z.shape))*torch.randn_like(z) if ts>0 else 0)
    img=vae.decode(z*scaling).clamp(-1,1)
    utils.save_image(utils.make_grid(0.5*(img+1),nrow=4,pad_value=1.0), root/'06-diffusion-models'/'_diag'/'e2e_samples.png')
print('saved e2e_samples.png')
