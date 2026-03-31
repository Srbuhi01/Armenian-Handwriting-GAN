"""
train.py  —  AC-WGAN-GP training for Mashtots Armenian handwriting dataset

Key changes vs original BCE-based AC-GAN:
  - Wasserstein loss + gradient penalty (WGAN-GP) for adversarial objective
    → fixes D_loss→0 / vanishing gradient problem observed in training logs
  - Critic trained n_critic=5 steps per generator step
  - Removed Sigmoid from D adversarial head (critic outputs raw score)
  - Label smoothing on real labels
  - Instance noise on real images fed to D (annealed to 0 over training)
  - Adam with betas=(0.0, 0.9) as recommended for WGAN-GP

Usage:
    python train.py --data_root ./data --epochs 300 --batch_size 64
    python train.py --data_root ./data --epochs 300 --batch_size 64 --resume ./checkpoints/checkpoint_epoch_0100.pt
"""

import os
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from src.dataset import get_dataloader, NUM_CLASSES
from src.model import Generator, Discriminator


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Gradient Penalty (WGAN-GP) ────────────────────────────────────────────────

def compute_gradient_penalty(D, real_imgs, fake_imgs, device):

    B = real_imgs.size(0)
    # Պատահական [0,1] միջակայքից ամեն  batch elemtի համար
    alpha = torch.rand(B, 1, 1, 1, device=device)
    interpolated = (alpha * real_imgs + (1 - alpha) * fake_imgs.detach()).requires_grad_(True)
    # requires_grad=True — PyTorch-ին ասում ենք "այս tensor-ի gradient-ը պետք է"

    # Critic-ը կիրառում ենք interpolated կետերի վրա
    d_interpolated, _ = D(interpolated)

    # Gradient հաշվում ենք D(interpolated)-ի նկատmամբ
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(B, -1)
    # Lipschitz պայման
    # Penalty = (||grad|| - 1)² → 0 երբ ||grad|| = 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Train] Using device: {device}")
    if device.type == 'cuda':
        print(f"[Train] GPU: {torch.cuda.get_device_name(0)}, "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Data ──────────────────────────────────────────────────────────────────
    loader = get_dataloader(
        root=args.data_root,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
    )

    # ── Models ────────────────────────────────────────────────────────────────
    G = Generator(
        num_classes=NUM_CLASSES,
        latent_dim=args.latent_dim,
        embed_dim=args.latent_dim,
        ngf=64,
    ).to(device)

    D = Discriminator(
        num_classes=NUM_CLASSES,
        ndf=64,
        wgan=True,   # removes Sigmoid — critic outputs raw score
    ).to(device)

    print(f"[Models] G params: {sum(p.numel() for p in G.parameters()):,}")
    print(f"[Models] D params: {sum(p.numel() for p in D.parameters()):,}")

    # ── Loss & Optimizers ─────────────────────────────────────────────────────
    cls_criterion = nn.CrossEntropyLoss()

    # WGAN-GP recommended: Adam with betas=(0.0, 0.9)
    opt_G = optim.Adam(G.parameters(), lr=args.lr, betas=(0.0, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=args.lr, betas=(0.0, 0.9))

    start_epoch = 1

    # ── Resume ────────────────────────────────────────────────────────────────
    if args.resume:
        # Նախորդում պահված checkpoint-ic շարունակում ենք ուսուցումը
        # G, D, ev optimizer-ների  state-երը
        ckpt = torch.load(args.resume, map_location=device)
        G.load_state_dict(ckpt['G_state_dict'])
        D.load_state_dict(ckpt['D_state_dict'])
        opt_G.load_state_dict(ckpt['opt_G'])
        opt_D.load_state_dict(ckpt['opt_D'])
        start_epoch = ckpt['epoch'] + 1
        print(f"[Resume] Continuing from epoch {start_epoch}")

    # ── Fixed noise for visualization ─────────────────────────────────────────
    os.makedirs(args.sample_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir,   exist_ok=True)

    fixed_noise  = torch.randn(NUM_CLASSES, args.latent_dim, device=device)
    fixed_labels = torch.arange(NUM_CLASSES, device=device)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        G.train(); D.train()

        # Instance noise std
        # Ուսուցման սկզբում աղմուկն ավելի մեծ է (regularization)
        # Ուսուցման ընթացքում աղմուկը նվազում է  e → 0
        noise_std = max(0.0, 0.1 * (1.0 - epoch / args.epochs))

        g_loss_epoch = d_loss_epoch = 0.0
        batches = 0

        data_iter = iter(loader)

        # We need enough batches for n_critic steps
        while True:
            # ── Train Critic n_critic times ───────────────────────────────────
            # WGAN-ը պետք է Critic-ին n_critic=5 անգամ սովորեցնի
            # մինչև Generator-ը 1 անգամ է սովորում
            # Այսպիսով Critic-ը դառնումէ ավելի լավը → Generator-ը ավելի լավ  gradient-ներ է ստաոնում
            d_loss_sum = 0.0
            critic_ok = True
            for _ in range(args.n_critic):
                try:
                    real_imgs, real_cls = next(data_iter)
                except StopIteration:
                    critic_ok = False
                    break

                B = real_imgs.size(0)
                real_imgs = real_imgs.to(device)
                real_cls  = real_cls.to(device)

                # Իրական նկարներին ավելացնում ենք փոքր աղմուկ

                if noise_std > 0:
                    real_imgs_noisy = real_imgs + noise_std * torch.randn_like(real_imgs)
                    real_imgs_noisy = real_imgs_noisy.clamp(-1, 1)
                else:
                    real_imgs_noisy = real_imgs

                opt_D.zero_grad()

                # Real
                real_validity, real_cls_pred = D(real_imgs_noisy)
                # Fake
                noise      = torch.randn(B, args.latent_dim, device=device)
                fake_labels = torch.randint(0, NUM_CLASSES, (B,), device=device)
                fake_imgs  = G(noise, fake_labels).detach()
                fake_validity, _ = D(fake_imgs)

                # Wasserstein adversarial loss
                w_loss = -real_validity.mean() + fake_validity.mean()

                # Gradient penalty
                gp = compute_gradient_penalty(D, real_imgs, fake_imgs, device)

                # Auxiliary classifier loss (real images only)
                cls_loss = cls_criterion(real_cls_pred, real_cls)

                d_loss = w_loss + args.lambda_gp * gp + args.cls_weight * cls_loss
                d_loss.backward()
                opt_D.step()
                d_loss_sum += d_loss.item()

            if not critic_ok:
                break

            # ── Train Generator, մեկ քայլով ──────────────────────────────────────────
            opt_G.zero_grad()

            noise      = torch.randn(B, args.latent_dim, device=device)
            fake_labels = torch.randint(0, NUM_CLASSES, (B,), device=device)
            fake_imgs  = G(noise, fake_labels)    #  Գեներատորը պետք է սովորի (թարմացնի կշիռները)

            g_validity, g_cls_pred = D(fake_imgs)

            # Գեներատորը ցանկանում է, որ (Critic/Discriminator)-ը  կեղծ պատկերներին հնարավորինս բարձր գնահատական տա
            # Այսինքն՝ Գեներատորն ուզում է, որ Դիսկրիմինատորն իր գեներացրած նկարին բարձր արժեք տա
            # Ինչը նշանակում է, որ D(fake) արժեքը պետք է մաքսիմալացնել
            g_adv = -g_validity.mean()
            # Գեներատորը նաև պետք է ճիշտ դասի (տառի) պատկերներ գեներացնի (դասակարգման կամ cls loss)
            g_cls = cls_criterion(g_cls_pred, fake_labels)

            g_loss = g_adv + args.cls_weight * g_cls
            g_loss.backward()
            opt_G.step()

            g_loss_epoch += g_loss.item()
            d_loss_epoch += d_loss_sum / args.n_critic
            batches += 1

        if batches == 0:
            continue

        # Ամեն էպոխի վերջում տպում ենք Դիսկրիմինատորի (D) և Գեներատորի (G) կորուստները (loss)
        # Եթե D_loss ≈ 0 և G_loss-ը կայուն է, նշանակում է մոդելը լավ է սովորում
        print(
            f"[Epoch {epoch:>3}/{args.epochs}]  "
            f"D_loss: {d_loss_epoch/batches:.4f}  "
            f"G_loss: {g_loss_epoch/batches:.4f}  "
            f"noise_std: {noise_std:.3f}"
        )

        # Պահպանել գեներացված նմուշների ցանցը (sample grid)
        if epoch % args.sample_every == 0 or epoch == args.epochs:
            # Բոլոր 78 տառերից մեկական նկար ենք գեներացնում ֆիքսված աղմուկով (fixed_noise)
            # Այսպիսով կարելի է տեսնել, թե մարզման ընթացքում ինչպես է բարելավվում տառերի որակը
            G.eval()
            with torch.no_grad():
                samples = G(fixed_noise, fixed_labels)
            samples = (samples + 1) / 2.0  # [-1,1] միջակայքից բերում ենք [0,1] միջակայքի՝ պատշաճ պահպանման/ցուցադրման համար
            save_image(
                samples,
                os.path.join(args.sample_dir, f"epoch_{epoch:04d}.png"),
                nrow=13,
                padding=2,
                normalize=False,
            )
            G.train()

        # Պահպանել մոդելի ընթացիկ վիճակը (checkpoint)
        if epoch % args.ckpt_every == 0 or epoch == args.epochs:
            # Պահպանում ենք Գեներատորի, Դիսկրիմինատորի և օպտիմիզատորների ամբողջական վիճակը
            # Որպեսզի հետագայում հնարավոր լինի շարունակել մարզումը հենց այս կետից (resume)
            torch.save({
                'epoch': epoch,
                'G_state_dict': G.state_dict(),
                'D_state_dict': D.state_dict(),
                'opt_G': opt_G.state_dict(),
                'opt_D': opt_D.state_dict(),
            }, os.path.join(args.ckpt_dir, f"checkpoint_epoch_{epoch:04d}.pt"))
            print(f"[Checkpoint] Saved at epoch {epoch}.")

    print("[Train] Done!")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train AC-WGAN-GP on Mashtots dataset')
    p.add_argument('--data_root',    type=str,   default='./data')
    p.add_argument('--epochs',       type=int,   default=300)
    p.add_argument('--batch_size',   type=int,   default=64)
    p.add_argument('--img_size',     type=int,   default=64)
    p.add_argument('--latent_dim',   type=int,   default=100)
    p.add_argument('--lr',           type=float, default=1e-4,    help='WGAN-GP recommended: 1e-4')
    p.add_argument('--lambda_gp',    type=float, default=10.0,    help='Gradient penalty weight')
    p.add_argument('--cls_weight',   type=float, default=1.0,     help='Auxiliary classifier loss weight')
    p.add_argument('--n_critic',     type=int,   default=5,       help='Critic steps per generator step')
    p.add_argument('--num_workers',  type=int,   default=2)
    p.add_argument('--sample_dir',   type=str,   default='./samples')
    p.add_argument('--ckpt_dir',     type=str,   default='./checkpoints')
    p.add_argument('--sample_every', type=int,   default=5)
    p.add_argument('--ckpt_every',   type=int,   default=20)
    p.add_argument('--resume',       type=str,   default=None,    help='Path to checkpoint to resume from')
    p.add_argument('--seed',         type=int,   default=42)
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())