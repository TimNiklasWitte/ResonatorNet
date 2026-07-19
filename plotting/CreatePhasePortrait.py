import numpy as np
import matplotlib.pyplot as plt

def main():

    # Parameters
    omega = 0.25
    damping = -0.20
    I = 0.5

    # Grid
    x = np.linspace(-3, 3, 30)
    y = np.linspace(-3, 3, 30)
    X, Y = np.meshgrid(x, y)

    # Vector field
    Xdot = damping * X - omega * Y + I
    Ydot = omega * X + damping * Y

    # Plot
    fig, ax = plt.subplots(figsize=(6,6))

    ax.streamplot(X, Y, Xdot, Ydot)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    
    plt.title(f"omega = {omega}\ndamping = {damping}\nI = {I}")
    plt.tight_layout()

    plt.savefig(f"./plots/PhasePortrait_omega_{omega}_damping_{damping}_I_{I}.png", dpi=200)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")