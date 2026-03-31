import pandas as pd
import matplotlib.pyplot as plt
import re
import os

# Գեղեցիկ սթայլ ակադեմիական գրաֆիկների համար
plt.style.use('seaborn-v0_8-whitegrid')


def smooth(scalars, weight=0.85):
    """Հարթեցնում է կորերը, որպեսզի վիզուալ մաքուր լինի:"""
    if len(scalars) == 0:
        return []
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def parse_acgan_log(txt_path):
    epochs, d_losses, g_losses = [], [], []
    if not os.path.exists(txt_path):
        print(f"Զգուշացում: {txt_path} ֆայլը չի գտնվել:")
        return pd.DataFrame()

    with open(txt_path, 'r') as file:
        lines = file.readlines()

    pattern = r"\[Epoch\s+(\d+)/\d+\]\s+D_loss:\s+([-\d\.]+)\s+G_loss:\s+([-\d\.]+)"
    for line in lines:
        match = re.search(pattern, line)
        if match:
            ep = int(match.group(1))
            epochs.append(ep)
            d_losses.append(float(match.group(2)))
            g_losses.append(float(match.group(3)))

    return pd.DataFrame({'Epoch': epochs, 'D_Loss': d_losses, 'G_Loss': g_losses})


def plot_fair_comparison():
    # 1. Բեռնում ենք տվյալները (որքան որ կան ֆայլերի մեջ)
    try:
        df_vae = pd.read_csv('../losses/vae_loss.csv')
        df_ddpm = pd.read_csv('../losses/ddpm_loss.csv')
        df_acgan = parse_acgan_log('../losses/acgan_log.txt')
    except Exception as e:
        print(f"❌ Խնդիր ֆայլերը կարդալիս. {e}")
        return

    # 2. Կառուցում ենք գրաֆիկները
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Գեներատիվ Մոդելների Ուսուցման Դինամիկան',
                 fontsize=16, fontweight='bold', y=1.05)

    # ---------------------------------------------------------
    # Գրաֆիկ 1: VAE
    # ---------------------------------------------------------
    if not df_vae.empty:
        axes[0].plot(df_vae.iloc[:, 0], df_vae.iloc[:, 1], alpha=0.3, color='blue')
        axes[0].plot(df_vae.iloc[:, 0], smooth(df_vae.iloc[:, 1].values), label='Smoothed Loss', color='blue', lw=2.5)
        axes[0].set_title('1. Baseline: Conditional VAE', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Դարաշրջան (Epoch)', fontsize=12)
        axes[0].set_ylabel('Reconstruction Loss (MSE)', fontsize=12)
        axes[0].legend()

    # ---------------------------------------------------------
    # Գրաֆիկ 2: DDPM
    # ---------------------------------------------------------
    if not df_ddpm.empty:
        axes[1].plot(df_ddpm.iloc[:, 0], df_ddpm.iloc[:, 1], alpha=0.3, color='green')
        axes[1].plot(df_ddpm.iloc[:, 0], smooth(df_ddpm.iloc[:, 1].values), label='Smoothed Noise Loss', color='green',
                     lw=2.5)
        axes[1].set_title('2. Baseline: DDPM (Diffusion)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Դարաշրջան (Epoch)', fontsize=12)
        axes[1].set_ylabel('Noise Prediction Loss', fontsize=12)
        axes[1].legend()

    # ---------------------------------------------------------
    # Գրաֆիկ 3: AC-WGAN-GP (300 Epochs)
    # ---------------------------------------------------------
    if not df_acgan.empty:
        axes[2].plot(df_acgan['Epoch'], df_acgan['D_Loss'], alpha=0.2, color='purple')
        axes[2].plot(df_acgan['Epoch'], smooth(df_acgan['D_Loss'].values), label='Critic (D) Loss', color='purple',
                     lw=2.5)

        axes[2].plot(df_acgan['Epoch'], df_acgan['G_Loss'], alpha=0.2, color='orange')
        axes[2].plot(df_acgan['Epoch'], smooth(df_acgan['G_Loss'].values), label='Generator (G) Loss', color='orange',
                     lw=2.5)

        axes[2].set_title('3. Proposed: AC-WGAN-GP', fontsize=14, fontweight='bold', color='darkred')
        axes[2].set_xlabel('Դարաշրջան (Epoch)', fontsize=12)
        axes[2].set_ylabel('Wasserstein Distance', fontsize=12)
        axes[2].legend()

    # Պահպանում ենք արդյունքը
    plt.tight_layout()
    output_img = 'clean_comparison_final.png'
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"✅ Մաքուր գրաֆիկը պահպանվեց '{output_img}' անունով!")
    plt.show()


if __name__ == '__main__':
    plot_fair_comparison()