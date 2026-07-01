"""Background DCGAN training that logs to wandb (mirrors GAN.ipynb)."""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, flush=True)

config = dict(image_size=64, channels=1, z_dim=100, g_features=64, d_features=64,
              batch_size=128, epochs=10, lr=2e-4, beta1=0.5, beta2=0.999, seed=42)
torch.manual_seed(config["seed"])

tf = transforms.Compose([transforms.Resize(config["image_size"]),
                         transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
ds = datasets.MNIST("./data", train=True, download=True, transform=tf)
dl = DataLoader(ds, batch_size=config["batch_size"], shuffle=True,
                num_workers=0, pin_memory=True, drop_last=True)


class Generator(nn.Module):
    def __init__(self, z, c, g):
        super().__init__()
        def b(i, o, k, s, p):
            return nn.Sequential(nn.ConvTranspose2d(i, o, k, s, p, bias=False),
                                 nn.BatchNorm2d(o), nn.ReLU(True))
        self.net = nn.Sequential(b(z, g*8, 4, 1, 0), b(g*8, g*4, 4, 2, 1),
                                 b(g*4, g*2, 4, 2, 1), b(g*2, g, 4, 2, 1),
                                 nn.ConvTranspose2d(g, c, 4, 2, 1), nn.Tanh())
    def forward(self, x): return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, c, d):
        super().__init__()
        def b(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1, bias=False),
                                 nn.BatchNorm2d(o), nn.LeakyReLU(0.2, True))
        self.net = nn.Sequential(nn.Conv2d(c, d, 4, 2, 1), nn.LeakyReLU(0.2, True),
                                 b(d, d*2), b(d*2, d*4), b(d*4, d*8),
                                 nn.Conv2d(d*8, 1, 4, 1, 0))
    def forward(self, x): return self.net(x).view(-1)


def weights_init(m):
    n = m.__class__.__name__
    if "Conv" in n:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in n:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


gen = Generator(config["z_dim"], config["channels"], config["g_features"]).to(device)
disc = Discriminator(config["channels"], config["d_features"]).to(device)
gen.apply(weights_init); disc.apply(weights_init)

criterion = nn.BCEWithLogitsLoss()
opt_gen = optim.Adam(gen.parameters(), config["lr"], betas=(config["beta1"], config["beta2"]))
opt_disc = optim.Adam(disc.parameters(), config["lr"], betas=(config["beta1"], config["beta2"]))
fixed_noise = torch.randn(64, config["z_dim"], 1, 1, device=device)

wandb.init(project="dcgan-mnist", config=config, name="dcgan-bg-run")

step = 0
for epoch in range(config["epochs"]):
    for real, _ in dl:
        real = real.to(device); bs = real.size(0)
        noise = torch.randn(bs, config["z_dim"], 1, 1, device=device)
        fake = gen(noise)
        dr = disc(real); df = disc(fake.detach())
        loss_d = (criterion(dr, torch.ones_like(dr)) + criterion(df, torch.zeros_like(df))) / 2
        opt_disc.zero_grad(); loss_d.backward(); opt_disc.step()
        out = disc(fake); loss_g = criterion(out, torch.ones_like(out))
        opt_gen.zero_grad(); loss_g.backward(); opt_gen.step()
        if step % 50 == 0:
            with torch.no_grad():
                D_x = torch.sigmoid(dr).mean().item(); D_Gz = torch.sigmoid(df).mean().item()
            wandb.log({"loss/discriminator": loss_d.item(), "loss/generator": loss_g.item(),
                       "D(x)": D_x, "D(G(z))": D_Gz, "epoch": epoch}, step=step)
        step += 1
    gen.eval()
    with torch.no_grad():
        samples = gen(fixed_noise)
    gen.train()
    grid = make_grid(samples, nrow=8, normalize=True, value_range=(-1, 1))
    wandb.log({"generated": wandb.Image(grid, caption=f"epoch {epoch+1}")}, step=step)
    save_image(grid, "latest_samples.png")
    print(f"epoch {epoch+1}/{config['epochs']} done  loss_d={loss_d.item():.3f} loss_g={loss_g.item():.3f}", flush=True)

os.makedirs("checkpoints", exist_ok=True)
torch.save(gen.state_dict(), "checkpoints/generator.pth")
torch.save(disc.state_dict(), "checkpoints/discriminator.pth")
wandb.finish()
print("TRAINING COMPLETE", flush=True)
