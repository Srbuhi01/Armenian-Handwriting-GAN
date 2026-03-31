import pandas as pd
import matplotlib.pyplot as plt


def plot_training_graphs(csv_path='vae_outputs/vae_loss.csv'):
    # Կարդում ենք պահպանված թվերը
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("CSV ֆայլը չի գտնվել: Նախ աշխատեցրեք baseline_cvae.py ֆայլը:")
        return

    epochs = df['Epoch']
    total_loss = df['Total_Loss']
    mse_loss = df['MSE_Loss']

    # Ստեղծում ենք գրաֆիկ
    plt.figure(figsize=(10, 6))

    # Գծում ենք Ընդհանուր Սխալը և Վերակառուցման Սխալը
    plt.plot(epochs, total_loss, label='Ընդհանուր Կորուստ (Total Loss)', color='blue', linewidth=2)
    plt.plot(epochs, mse_loss, label='Վերակառուցման Կորուստ (MSE Loss)', color='red', linestyle='dashed')

    # Դիզայն (Ակադեմիական ստանդարտ)
    plt.title('Conditional VAE Ուսուցման Դինամիկան (Baseline Model)', fontsize=14, fontweight='bold')
    plt.xlabel('Դարաշրջան (Epochs)', fontsize=12)
    plt.ylabel('Կորուստ (Loss)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    # Պահպանում ենք նկարը
    plt.tight_layout()
    plt.savefig('vae_outputs/loss_graph.png', dpi=300)
    print("✅ Գրաֆիկը պահպանվեց 'vae_outputs/loss_graph.png' անունով:")

    plt.show()


if __name__ == '__main__':
    plot_training_graphs()