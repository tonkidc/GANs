import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

tf = transforms.Compose([transforms.Resize(64), transforms.ToTensor(), transforms.Normalize([0.5],[0.5])])
ds = datasets.MNIST(root="./data", train=True, download=True, transform=tf)
dl = DataLoader(ds, batch_size=128, shuffle=True, num_workers=0, drop_last=True)
print("batches:", len(dl))

class G(nn.Module):
    def __init__(s):
        super().__init__()
        def b(i,o,k,st,p): return nn.Sequential(nn.ConvTranspose2d(i,o,k,st,p,bias=False),nn.BatchNorm2d(o),nn.ReLU(True))
        s.net=nn.Sequential(b(100,512,4,1,0),b(512,256,4,2,1),b(256,128,4,2,1),b(128,64,4,2,1),nn.ConvTranspose2d(64,1,4,2,1),nn.Tanh())
    def forward(s,x): return s.net(x)
class D(nn.Module):
    def __init__(s):
        super().__init__()
        def b(i,o): return nn.Sequential(nn.Conv2d(i,o,4,2,1,bias=False),nn.BatchNorm2d(o),nn.LeakyReLU(0.2,True))
        s.net=nn.Sequential(nn.Conv2d(1,64,4,2,1),nn.LeakyReLU(0.2,True),b(64,128),b(128,256),b(256,512),nn.Conv2d(512,1,4,1,0))
    def forward(s,x): return s.net(x).view(-1)

g=G().to(device); d=D().to(device)
crit=nn.BCEWithLogitsLoss()
og=optim.Adam(g.parameters(),2e-4,betas=(0.5,0.999)); od=optim.Adam(d.parameters(),2e-4,betas=(0.5,0.999))

print("running 30 training steps...")
it=iter(dl)
for step in range(30):
    real,_=next(it); real=real.to(device); bs=real.size(0)
    noise=torch.randn(bs,100,1,1,device=device); fake=g(noise)
    dr=d(real); df=d(fake.detach())
    ld=(crit(dr,torch.ones_like(dr))+crit(df,torch.zeros_like(df)))/2
    od.zero_grad(); ld.backward(); od.step()
    out=d(fake); lg=crit(out,torch.ones_like(out))
    og.zero_grad(); lg.backward(); og.step()
    if step%10==0: print(f"  step {step}: loss_d={ld.item():.3f} loss_g={lg.item():.3f}")

# save a sample image to confirm generation works
with torch.no_grad(): samp=g(torch.randn(64,100,1,1,device=device))
grid=make_grid(samp,nrow=8,normalize=True,value_range=(-1,1)).cpu()
from torchvision.utils import save_image
save_image(grid,"_smoketest_sample.png")
print("SUCCESS: saved _smoketest_sample.png — code works end to end on GPU")
