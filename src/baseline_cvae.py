import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torchvision.utils import save_image
from tqdm import tqdm  # Ավելացված է վիզուալիզացիայի համար
from dataset import get_dataloader, NUM_CLASSES


# ==========================================
# 1. ՃԱՐՏԱՐԱՊԵՏՈՒԹՅՈՒՆ: Conditional VAE (cVAE)
# ==========================================
class cVAE(nn.Module):
    def __init__(self, latent_dim=100, num_classes=NUM_CLASSES):
        super(cVAE, self).__init__()
        self.latent_dim = latent_dim

        # Դասերի Embedding (Տառի ինդեքսը -> վեկտոր)
        self.class_emb = nn.Embedding(num_classes, 50)

        # ԷՆԿՈԴԵՐ (Encoder) - Նկարը սեղմում է
        # Մուտք: 1 ալիք (նկար) + 50 ալիք (դասը վերածած մատրիցի) = 51 ալիք
        self.encoder = nn.Sequential(
            nn.Conv2d(1 + 50, 32, 4, 2, 1), nn.LeakyReLU(0.2),  # 32x32
            nn.Conv2d(32, 64, 4, 2, 1), nn.LeakyReLU(0.2),  # 16x16
            nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2),  # 8x8
            nn.Conv2d(128, 256, 4, 2, 1), nn.LeakyReLU(0.2),  # 4x4
            nn.Flatten()  # 256 * 4 * 4 = 4096
        )

        # Լատենտ տարածության վեկտորներ (Mu և LogVar)
        self.fc_mu = nn.Linear(4096, latent_dim)
        self.fc_logvar = nn.Linear(4096, latent_dim)

        # ԴԵԿՈԴԵՐ (Decoder) - Սեղմածից նորից նկար է սարքում
        self.decoder_input = nn.Linear(latent_dim + 50, 4096)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Tanh()  # Ելքը [-1, 1] միջակայքում
        )

    def encode(self, x, labels):
        emb = self.class_emb(labels).unsqueeze(2).unsqueeze(3)
        emb = emb.expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, emb], dim=1)  # Նկարին կպցնում ենք տառի ինֆոն
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, labels):
        emb = self.class_emb(labels)
        z = torch.cat([z, emb], dim=1)
        h = self.decoder_input(z).view(-1, 256, 4, 4)
        return self.decoder(h)

    def forward(self, x, labels):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z, labels)
        return reconstructed, mu, logvar


# ==========================================
# 2. VAE LOSS ՖՈՒՆԿՑԻԱ (MSE + KL Divergence)
# ==========================================
def vae_loss_function(recon_x, x, mu, logvar):
    # Վերակառուցման սխալ (որքանով է նման օրիգինալին)
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    # KL շեղում (Լատենտ տարածության կանոնավորում)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD, MSE, KLD


# ==========================================
# 3. ՄԱՐԶՈՒՄ ԵՎ ՏՎՅԱԼՆԵՐԻ ՊԱՀՊԱՆՈՒՄ (Logging)
# ==========================================
def train_cvae():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Մարզում ենք VAE մոդելը: Սարքավորում՝ {device}")

    # Պարամետրեր
    epochs = 100  # VAE-ին 50-ն էլ է հերիք (հատկապես baseline ցույց տալու համար)
    batch_size = 64
    learning_rate = 1e-3

    loader = get_dataloader(root='./data', batch_size=batch_size, img_size=64)
    model = cVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    os.makedirs('../vae_outputs', exist_ok=True)

    # Գրաֆիկների համար տվյալներ հավաքելու ցուցակ
    loss_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0;
        train_mse = 0;
        train_kld = 0

        # tqdm-ը կստեղծի դինամիկ առաջընթացի գիծ տերմինալում
        progress_bar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", unit="batch", leave=False)

        for imgs, labels in progress_bar:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            recon_imgs, mu, logvar = model(imgs, labels)

            loss, mse, kld = vae_loss_function(recon_imgs, imgs, mu, logvar)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_mse += mse.item()
            train_kld += kld.item()

            # Թարմացնում ենք tqdm-ի վրա երևացող թվերը (loss)
            progress_bar.set_postfix({'Loss': loss.item() / imgs.size(0)})

        # Հաշվում ենք միջին սխալները այս դարաշրջանի համար
        avg_loss = train_loss / len(loader.dataset)
        avg_mse = train_mse / len(loader.dataset)
        avg_kld = train_kld / len(loader.dataset)

        # Տպում ենք ամեն էպոխի վերջնական արդյունքը
        print(
            f"Epoch [{epoch}/{epochs}] Ավարտված | Ընդհ. Loss: {avg_loss:.4f} | MSE: {avg_mse:.4f} | KLD: {avg_kld:.4f}")

        # Պահպանում ենք թվերը գրաֆիկի համար
        loss_history.append({'Epoch': epoch, 'Total_Loss': avg_loss, 'MSE_Loss': avg_mse, 'KL_Loss': avg_kld})

        # Ամեն 10 փուլը մեկ պահպանում ենք նկար որպես ապացույց
        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                # ՆՈՐՈՒՅԹ. Ստեղծում ենք ճիշտ 78 աղմուկ և 0-ից 77 բոլոր դասերը հերթականությամբ
                # Սա ապահովում է իդեալական "Fair Comparison" մյուս մոդելի հետ
                z = torch.randn(NUM_CLASSES, 100).to(device)
                sample_labels = torch.arange(0, NUM_CLASSES).to(device)

                sample = model.decode(z, sample_labels)
                sample = (sample + 1) / 2.0  # Վերադարձնում ենք [0, 1] պիքսելային տիրույթ

                # nrow=13 ապահովում է 13 սյունակ x 6 տող (ճիշտ այնպես, ինչպես GAN-ում)
                save_image(sample, f'vae_outputs/sample_epoch_{epoch}.png', nrow=13)

    # Մարզման ավարտից հետո պահպանում ենք CSV ֆայլը (Գրաֆիկների համար)
    df = pd.DataFrame(loss_history)
    df.to_csv('vae_outputs/vae_loss.csv', index=False)
    print("✅ VAE մարզումն ավարտված է! Տվյալները պահպանվեցին 'vae_outputs/vae_loss.csv' ֆայլում:")


if __name__ == '__main__':
    train_cvae()