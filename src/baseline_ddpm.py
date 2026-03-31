import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd  # ՆՈՐՈՒՅԹ. Տվյալների պահպանման համար
from torchvision.utils import save_image
from tqdm import tqdm
from dataset import get_dataloader, NUM_CLASSES


# ==========================================
# 1. ԴԻՖՈՒԶԻՈՆ ՄՈԴԵԼԻ ՃԱՐՏԱՐԱՊԵՏՈՒԹՅՈՒՆ (Mini U-Net)
# ==========================================
class MiniUNet(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=64, num_classes=NUM_CLASSES):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        self.class_emb = nn.Embedding(num_classes, time_dim)

        self.down1 = nn.Conv2d(c_in, 32, 4, 2, 1)
        self.down2 = nn.Conv2d(32, 64, 4, 2, 1)
        self.mid1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.up1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.up2 = nn.ConvTranspose2d(32, c_out, 4, 2, 1)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, t, labels):
        t = t.unsqueeze(-1).type(torch.float)
        t_emb = self.time_mlp(t)
        c_emb = self.class_emb(labels)
        emb = (t_emb + c_emb).unsqueeze(-1).unsqueeze(-1)

        d1 = self.act(self.down1(x))
        d2 = self.act(self.down2(d1))
        m = self.act(self.mid1(d2 + emb))
        u1 = self.act(self.up1(m))
        return self.up2(u1)


# ==========================================
# 2. ԴԻՖՈՒԶԻՈՆ ՊՐՈՑԵՍ (DDPM Schedule)
# ==========================================
class Diffusion:
    def __init__(self, timesteps=250, device='cuda'):
        self.timesteps = timesteps
        self.device = device
        self.beta = torch.linspace(1e-4, 0.02, timesteps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.timesteps, size=(n,)).to(self.device)

    @torch.no_grad()
    def sample(self, model, n, labels):
        model.eval()
        x = torch.randn((n, 1, 64, 64)).to(self.device)
        for i in reversed(range(1, self.timesteps)):
            t = (torch.ones(n) * i).long().to(self.device)
            predicted_noise = model(x, t, labels)
            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            beta = self.beta[t][:, None, None, None]
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            x = 1 / torch.sqrt(alpha) * (
                        x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        return x


# ==========================================
# 3. ՄԱՐԶՈՒՄ ԵՎ ԳՐԱՖԻԿՆԵՐԻ ՏՎՅԱԼՆԵՐ (Training & Logging)
# ==========================================
def train_ddpm():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Մարզում ենք DDPM (Diffusion) մոդելը: Սարքավորում՝ {device}")

    epochs = 100
    batch_size = 64
    loader = get_dataloader(root='./data', batch_size=batch_size, img_size=64)
    model = MiniUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    mse_loss = nn.MSELoss()
    diffusion = Diffusion(timesteps=250, device=device)

    os.makedirs('../ddpm_outputs', exist_ok=True)

    # ՆՈՐՈՒՅԹ. Ստեղծում ենք ցուցակ գրաֆիկների թվերը պահելու համար
    loss_history = []

    for epoch in range(1, epochs + 1):
        progress_bar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", unit="batch", leave=False)
        epoch_loss = 0

        for imgs, labels in progress_bar:
            imgs, labels = imgs.to(device), labels.to(device)
            t = diffusion.sample_timesteps(imgs.shape[0])
            x_t, noise = diffusion.noise_images(imgs, t)
            predicted_noise = model(x_t, t, labels)
            loss = mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'Noise MSE Loss': loss.item()})

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch [{epoch}/{epochs}] Ավարտված | Միջին Noise Loss: {avg_loss:.4f}")

        # Պահպանում ենք պատմությունը
        loss_history.append({'Epoch': epoch, 'Noise_MSE_Loss': avg_loss})

        if epoch == epochs:
            print("⏳ Սկսվում է Դիֆուզիոն գեներացիան (250 քայլ). Սա դանդաղ է...")
            # ՃԻՇՏ ՆՈՒՅՆ ՉԱՓԸ ԻՆՉ VAE-Ն ԵՎ WGAN-Ը (78 տառեր)
            labels = torch.arange(0, NUM_CLASSES).to(device)
            sampled_images = diffusion.sample(model, n=NUM_CLASSES, labels=labels)

            # nrow=13 ապահովում է 13x6 ցանց (Grid)
            save_image(sampled_images, '../ddpm_outputs/ddpm_result.png', nrow=13)
            print("✅ DDPM Նկարը պահպանված է 'ddpm_outputs/ddpm_result.png':")

    # ՆՈՐՈՒՅԹ. Պահպանում ենք CSV ֆայլը ապագա գրաֆիկների համար
    df = pd.DataFrame(loss_history)
    df.to_csv('ddpm_outputs/vae_loss.csv', index=False)
    print("✅ Տվյալները գրաֆիկի համար պահպանվեցին 'ddpm_outputs/vae_loss.csv':")


if __name__ == '__main__':
    train_ddpm()