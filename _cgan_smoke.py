import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
device = torch.device('cuda')
tf = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
dl = DataLoader(datasets.MNIST('./data', True, download=True, transform=tf), 128, shuffle=True, num_workers=0, drop_last=True)


class G(nn.Module):
    def __init__(s, z=100, c=1, g=64, nc=10, es=100):
        super().__init__(); s.embed = nn.Embedding(nc, es)
        def b(i, o, k, st, p): return nn.Sequential(nn.ConvTranspose2d(i, o, k, st, p, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))
        s.net = nn.Sequential(b(z+es, g*4, 4, 1, 0), b(g*4, g*2, 4, 2, 1), b(g*2, g, 4, 2, 1), nn.ConvTranspose2d(g, c, 4, 2, 1), nn.Tanh())
    def forward(s, n, l):
        e = s.embed(l).unsqueeze(-1).unsqueeze(-1); return s.net(torch.cat([n, e], 1))


class D(nn.Module):
    def __init__(s, c=1, d=64, nc=10, isz=32):
        super().__init__(); s.isz = isz; s.embed = nn.Embedding(nc, isz*isz)
        def b(i, o): return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1, bias=False), nn.BatchNorm2d(o), nn.LeakyReLU(0.2, True))
        s.net = nn.Sequential(nn.Conv2d(c+1, d, 4, 2, 1), nn.LeakyReLU(0.2, True), b(d, d*2), b(d*2, d*4), nn.Conv2d(d*4, 1, 4, 1, 0))
    def forward(s, x, l):
        e = s.embed(l).view(l.size(0), 1, s.isz, s.isz); return s.net(torch.cat([x, e], 1)).view(-1)


g = G().to(device); d = D().to(device); crit = nn.BCEWithLogitsLoss()
og = optim.Adam(g.parameters(), 2e-4, betas=(0.5, 0.999)); od = optim.Adam(d.parameters(), 2e-4, betas=(0.5, 0.999))
it = iter(dl)
for stepi in range(15):
    real, lab = next(it); real = real.to(device); lab = lab.to(device); bs = real.size(0)
    noise = torch.randn(bs, 100, 1, 1, device=device); fake = g(noise, lab)
    dr = d(real, lab); df = d(fake.detach(), lab); ld = (crit(dr, torch.ones_like(dr)) + crit(df, torch.zeros_like(df)))/2
    od.zero_grad(); ld.backward(); od.step()
    out = d(fake, lab); lg = crit(out, torch.ones_like(out)); og.zero_grad(); lg.backward(); og.step()
    if stepi % 5 == 0:
        print(f'  step {stepi}: loss_d={ld.item():.3f} loss_g={lg.item():.3f} | fake shape {tuple(fake.shape)}', flush=True)
with torch.no_grad():
    s = g(torch.randn(8, 100, 1, 1, device=device), torch.full((8,), 7, dtype=torch.long, device=device))
print('SUCCESS: cGAN works, generated 8 sevens, shape', tuple(s.shape), flush=True)
