import torch, torch.nn as nn, torch.optim as optim, time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
device=torch.device("cuda")
tf=transforms.Compose([transforms.Resize(32),transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])
dl=DataLoader(datasets.MNIST("./data",True,download=True,transform=tf),64,shuffle=True,num_workers=0,drop_last=True,pin_memory=True)
def gb(i,o,k,st,p): return nn.Sequential(nn.ConvTranspose2d(i,o,k,st,p,bias=False),nn.BatchNorm2d(o),nn.ReLU(True))
g=nn.Sequential(gb(100,256,4,1,0),gb(256,128,4,2,1),gb(128,64,4,2,1),nn.ConvTranspose2d(64,1,4,2,1),nn.Tanh()).to(device)
def db(i,o): return nn.Sequential(nn.Conv2d(i,o,4,2,1,bias=False),nn.InstanceNorm2d(o,affine=True),nn.LeakyReLU(0.2,True))
class C(nn.Module):
    def __init__(s): super().__init__(); s.n=nn.Sequential(nn.Conv2d(1,64,4,2,1),nn.LeakyReLU(0.2,True),db(64,128),db(128,256),nn.Conv2d(256,1,4,1,0))
    def forward(s,x): return s.n(x).view(-1)
c=C().to(device)
og=optim.Adam(g.parameters(),1e-4,betas=(0.0,0.9)); oc=optim.Adam(c.parameters(),1e-4,betas=(0.0,0.9))
def gp(crit,real,fake):
    bs=real.size(0); eps=torch.rand(bs,1,1,1,device=device).expand_as(real)
    inter=(eps*real+(1-eps)*fake).requires_grad_(True); sc=crit(inter)
    gr=torch.autograd.grad(sc,inter,torch.ones_like(sc),create_graph=True,retain_graph=True)[0].view(bs,-1)
    return ((gr.norm(2,dim=1)-1)**2).mean()
it=iter(dl)
# warmup 5
for _ in range(5):
    real,_=next(it); real=real.to(device); bs=real.size(0)
    for _ in range(5):
        fake=g(torch.randn(bs,100,1,1,device=device)); lc=-(c(real).mean()-c(fake.detach()).mean())+10*gp(c,real,fake.detach())
        oc.zero_grad(); lc.backward(); oc.step()
    fake=g(torch.randn(bs,100,1,1,device=device)); lg=-c(fake).mean(); og.zero_grad(); lg.backward(); og.step()
N=100; t=time.time()
for _ in range(N):
    real,_=next(it); real=real.to(device); bs=real.size(0)
    for _ in range(5):
        fake=g(torch.randn(bs,100,1,1,device=device)); lc=-(c(real).mean()-c(fake.detach()).mean())+10*gp(c,real,fake.detach())
        oc.zero_grad(); lc.backward(); oc.step()
    fake=g(torch.randn(bs,100,1,1,device=device)); lg=-c(fake).mean(); og.zero_grad(); lg.backward(); og.step()
el=time.time()-t
per=el/N
print(f"warm speed: {per*1000:.0f} ms/batch -> 1 epoch (937 batches) = {per*937/60:.1f} min -> 10 epochs = {per*937*10/60:.0f} min")
