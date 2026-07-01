import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
tf=transforms.Compose([transforms.Resize(32),transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])
dl=DataLoader(datasets.MNIST("./data",True,download=True,transform=tf),64,shuffle=True,num_workers=0,drop_last=True)

class G(nn.Module):
    def __init__(s,z=100,c=1,g=64):
        super().__init__()
        def b(i,o,k,st,p): return nn.Sequential(nn.ConvTranspose2d(i,o,k,st,p,bias=False),nn.BatchNorm2d(o),nn.ReLU(True))
        s.net=nn.Sequential(b(z,g*4,4,1,0),b(g*4,g*2,4,2,1),b(g*2,g,4,2,1),nn.ConvTranspose2d(g,c,4,2,1),nn.Tanh())
    def forward(s,x): return s.net(x)
class C(nn.Module):
    def __init__(s,c=1,d=64):
        super().__init__()
        def b(i,o): return nn.Sequential(nn.Conv2d(i,o,4,2,1,bias=False),nn.InstanceNorm2d(o,affine=True),nn.LeakyReLU(0.2,True))
        s.net=nn.Sequential(nn.Conv2d(c,d,4,2,1),nn.LeakyReLU(0.2,True),b(d,d*2),b(d*2,d*4),nn.Conv2d(d*4,1,4,1,0))
    def forward(s,x): return s.net(x).view(-1)

def gp(crit,real,fake):
    bs=real.size(0)
    eps=torch.rand(bs,1,1,1,device=device).expand_as(real)
    inter=(eps*real+(1-eps)*fake).requires_grad_(True)
    sc=crit(inter)
    gr=torch.autograd.grad(sc,inter,torch.ones_like(sc),create_graph=True,retain_graph=True)[0]
    gr=gr.view(bs,-1)
    return ((gr.norm(2,dim=1)-1)**2).mean()

g=G().to(device); c=C().to(device)
og=optim.Adam(g.parameters(),1e-4,betas=(0.0,0.9)); oc=optim.Adam(c.parameters(),1e-4,betas=(0.0,0.9))
print("running 20 steps with gradient penalty (double-backward)...")
it=iter(dl)
for step in range(20):
    real,_=next(it); real=real.to(device); bs=real.size(0)
    for _ in range(5):
        noise=torch.randn(bs,100,1,1,device=device); fake=g(noise)
        cr=c(real); cf=c(fake.detach()); penalty=gp(c,real,fake.detach())
        lc=-(cr.mean()-cf.mean())+10*penalty
        oc.zero_grad(); lc.backward(); oc.step()
    noise=torch.randn(bs,100,1,1,device=device); fake=g(noise)
    lg=-c(fake).mean(); og.zero_grad(); lg.backward(); og.step()
    if step%5==0: print(f"  step {step}: loss_c={lc.item():.3f} loss_g={lg.item():.3f} W_dist={(cr.mean()-cf.mean()).item():.3f} gp={penalty.item():.3f}")
print("SUCCESS: WGAN-GP gradient penalty works on this GPU")
