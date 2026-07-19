import sys
sys.path.append("./..")

import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from Classifier import *

NUM_EPOCHS = 16

def main():
    model = Classifier()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(NUM_EPOCHS):

        checkpoint = torch.load(f"./../saved_models/{epoch}", map_location=device)
        model.load_state_dict(checkpoint)

        # Extract parameters
        omega_hidden = model.raf_hidden.omega.detach().cpu().numpy()
        damping_hidden = model.raf_hidden.damping.detach().cpu().numpy()
        omega_hidden = omega_hidden.flatten()
        damping_hidden = damping_hidden.flatten()

        damping_hidden = -np.abs(damping_hidden)

        omega_output = model.raf_output.omega.detach().cpu().numpy()
        damping_output = model.raf_output.damping.detach().cpu().numpy()
        omega_output = omega_output.flatten()
        damping_output = damping_output.flatten()

        damping_output = -np.abs(damping_output)

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

        #
        # Subplot 1
        #

        sns.kdeplot(
                x=omega_hidden,
                y=damping_hidden,
                ax=axes[0],
                cmap="Blues",
                fill=True,
                alpha=0.6,
                levels=5
            )
        axes[0].scatter(omega_hidden, damping_hidden, alpha=0.3, s=20, c='blue', edgecolors='black', linewidth=0.5)
        axes[0].set_xlabel('Omega', fontsize=12)
        axes[0].set_ylabel('Damping', fontsize=12)
        axes[0].set_title('Hidden Layer: Omega vs Damping', fontsize=14)
        axes[0].grid(True, alpha=0.3)


        axes[0].set_xlim(0, 0.5)
        axes[0].set_ylim(-0.3, 0.01)


        #
        # Subplot 2
        #

        sns.kdeplot(
                x=omega_output,
                y=damping_output,
                ax=axes[1],
                cmap="Greens",
                fill=True,
                alpha=0.6,
                levels=5
        )
        axes[1].scatter(omega_output, damping_output, alpha=0.3, s=20, c='green', edgecolors='black', linewidth=0.5)
        axes[1].set_xlabel('Omega', fontsize=12)
        axes[1].set_ylabel('Damping', fontsize=12)
        axes[1].set_title('Output Layer: Omega vs Damping', fontsize=14)
        axes[1].grid(True, alpha=0.3)

        axes[1].set_xlim(0, 0.5)
        axes[1].set_ylim(-0.3, 0.01)

        plt.suptitle(f"Epoch: {epoch}", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"./plots/KDE_Epoch_{epoch}.png", dpi=200)
        plt.close()
 

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")