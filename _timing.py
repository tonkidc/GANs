import torch, torch.nn as nn, torch.optim as optim, time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
device=torch.device("cuda")
tf=transforms.Compose([transforms.Resize(64),transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])
dl=DataLoader(datasets.MNIST("./data",True,download=True,transform=tf),128,shuffle=True,num_workers=0,drop_last=True,pin_memory=True)
def gb(i,o,k,s,p): return nn.Sequential(nn.ConvTranspose2d(i,o,k,s,p,bias=False),nn.BatchNorm2d(o),nn.ReLU(True))
def db(i,o): return nn.Sequential(nn.Conv2d(i,o,4,2,1,bias=False),nn.BatchNorm2d(o),nn.LeakyReLU(0.2,True))
g=nn.Sequential(gb(100,512,4,1,0),gb(512,256,4,2,1),gb(256,128,4,2,1),gb(128,64,4,2,1),nn.ConvTranspose2d(64,1,4,2,1),nn.Tanh()).to(device)
class D(nn.Module):
    def __init__(s): super().__init__(); s.n=nn.Sequential(nn.Conv2d(1,64,4,2,1),nn.LeakyReLU(0.2,True),db(64,128),db(128,256),db(256,512),nn.Conv2d(512,1,4,1,0))
    def forward(s,x): return s.n(x).view(-1)
d=D().to(device); crit=nn.BCEWithLogitsLoss()
og=optim.Adam(g.parameters(),2e-4,betas=(0.5,0.999)); od=optim.Adam(d.parameters(),2e-4,betas=(0.5,0.999))
t=time.time(); n=0
for real,_ in dl:
    real=real.to(device); bs=real.size(0); noise=torch.randn(bs,100,1,1,device=device); fake=g(noise)
    dr=d(real); df=d(fake.detach()); ld=(crit(dr,torch.ones_like(dr))+crit(df,torch.zeros_like(df)))/2
    od.zero_grad(); ld.backward(); od.step()
    out=d(fake); lg=crit(out,torch.ones_like(out)); og.zero_grad(); lg.backward(); og.step(); n+=1
el=time.time()-t
print(f"1 epoch = {el:.1f}s over {n} batches -> 20 epochs ~ {el*20/60:.1f} min")
