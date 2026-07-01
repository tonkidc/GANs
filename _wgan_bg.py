"""Background WGAN-GP training (mirrors WGAN_GP.ipynb), logs to wandb."""
import os, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, flush=True)
cfg = dict(image_size=32, channels=1, z_dim=100, g_features=64, d_features=64,
           batch_size=64, epochs=10, lr=1e-4, beta1=0.0, beta2=0.9, n_critic=5,
           lambda_gp=10, seed=42)
torch.manual_seed(cfg["seed"])

tf = transforms.Compose([transforms.Resize(cfg["image_size"]), transforms.ToTensor(),
                         transforms.Normalize([0.5], [0.5])])
dl = DataLoader(datasets.MNIST("./data", True, download=True, transform=tf),
                cfg["batch_size"], shuffle=True, num_workers=0, pin_memory=True, drop_last=True)


class G(nn.Module):
    def __init__(s, z, c, g):
        super().__init__()
        def b(i, o, k, st, p):
            return nn.Sequential(nn.ConvTranspose2d(i, o, k, st, p, bias=False),
                                 nn.BatchNorm2d(o), nn.ReLU(True))
        s.net = nn.Sequential(b(z, g*4, 4, 1, 0), b(g*4, g*2, 4, 2, 1),
                              b(g*2, g, 4, 2, 1), nn.ConvTranspose2d(g, c, 4, 2, 1), nn.Tanh())
    def forward(s, x): return s.net(x)


class C(nn.Module):
    def __init__(s, c, d):
        super().__init__()
        def b(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1, bias=False),
                                 nn.InstanceNorm2d(o, affine=True), nn.LeakyReLU(0.2, True))
        s.net = nn.Sequential(nn.Conv2d(c, d, 4, 2, 1), nn.LeakyReLU(0.2, True),
                              b(d, d*2), b(d*2, d*4), nn.Conv2d(d*4, 1, 4, 1, 0))
    def forward(s, x): return s.net(x).view(-1)


def winit(m):
    n = m.__class__.__name__
    if "Conv" in n:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in n:
        nn.init.normal_(m.weight.data, 1.0, 0.02); nn.init.constant_(m.bias.data, 0)


def gp(crit, real, fake):
    bs = real.size(0)
    eps = torch.rand(bs, 1, 1, 1, device=device).expand_as(real)
    inter = (eps*real + (1-eps)*fake).requires_grad_(True)
    sc = crit(inter)
    gr = torch.autograd.grad(sc, inter, torch.ones_like(sc), create_graph=True, retain_graph=True)[0]
    gr = gr.view(bs, -1)
    return ((gr.norm(2, dim=1) - 1)**2).mean()


gen = G(cfg["z_dim"], cfg["channels"], cfg["g_features"]).to(device); gen.apply(winit)
crit = C(cfg["channels"], cfg["d_features"]).to(device); crit.apply(winit)
og = optim.Adam(gen.parameters(), cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
oc = optim.Adam(crit.parameters(), cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
fixed = torch.randn(64, cfg["z_dim"], 1, 1, device=device)

wandb.init(project="wgan-gp-mnist", config=cfg, name="wgan-gp-bg-run")
step = 0
for epoch in range(cfg["epochs"]):
    for real, _ in dl:
        real = real.to(device); bs = real.size(0)
        for _ in range(cfg["n_critic"]):
            noise = torch.randn(bs, cfg["z_dim"], 1, 1, device=device); fake = gen(noise)
            cr = crit(real); cf = crit(fake.detach()); pen = gp(crit, real, fake.detach())
            lc = -(cr.mean() - cf.mean()) + cfg["lambda_gp"]*pen
            oc.zero_grad(); lc.backward(); oc.step()
        noise = torch.randn(bs, cfg["z_dim"], 1, 1, device=device); fake = gen(noise)
        lg = -crit(fake).mean(); og.zero_grad(); lg.backward(); og.step()
        if step % 50 == 0:
            wdist = (cr.mean() - cf.mean()).item()
            wandb.log({"loss/critic": lc.item(), "loss/generator": lg.item(),
                       "wasserstein_distance": wdist, "gradient_penalty": pen.item(),
                       "epoch": epoch}, step=step)
        step += 1
    gen.eval()
    with torch.no_grad():
        s = gen(fixed)
    gen.train()
    grid = make_grid(s, nrow=8, normalize=True, value_range=(-1, 1))
    wandb.log({"generated": wandb.Image(grid, caption=f"epoch {epoch+1}")}, step=step)
    save_image(grid, "wgan_latest_samples.png")
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(gen.state_dict(), f"checkpoints/wgan_gen_ep{epoch+1}.pth")
    print(f"epoch {epoch+1}/{cfg['epochs']} done  loss_c={lc.item():.2f} W_dist={(cr.mean()-cf.mean()).item():.2f}", flush=True)

torch.save(gen.state_dict(), "checkpoints/wgan_generator.pth")
wandb.finish()
print("WGAN-GP TRAINING COMPLETE", flush=True)
