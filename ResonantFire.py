import torch
import torch.nn as nn


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, threshold):
        ctx.save_for_backward(v)
        ctx.threshold = threshold

        return (v >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        (v,) = ctx.saved_tensors

        x = v - ctx.threshold
        k = 1

        grad = 1.0 / (1.0 + (k * x.abs()) ** 2)

        return grad_output * grad, None


class ResonantFire(nn.Module):
    """
    Resonant-and-Fire neuron.

    State:
        x : membrane variable
        y : resonant variable
    """

    def __init__(
        self,
        n,
        omega=0.2,
        damping=-0.05,
        reset_x=0.0,
        reset_y=0.0,
        threshold=1.0,
    ):
        super().__init__()

        self.omega = nn.Parameter(
            torch.full((n,), omega),
            requires_grad=True
        )

        self.damping = nn.Parameter(
            torch.full((n,), damping),
            requires_grad=True
        )

        self.reset_x = nn.Parameter(
            torch.full((n,), reset_x),
            requires_grad=False
        )

        self.reset_y = nn.Parameter(
            torch.full((n,), reset_y),
            requires_grad=False
        )

        self.threshold = threshold

    def forward(self, x, y, I, dt=1.0):
        """
        x : membrane state
        y : resonant state
        I : input current
        """

        # Damped oscillator dynamics
        dx = self.damping * x - self.omega * y + I
        dy = self.omega * x + self.damping * y

        # Euler integration
        x = x + dt * dx
        y = y + dt * dy

        # Spike generation
        spikes = SurrogateSpike.apply(
            x,
            self.threshold
        )

        # Reset after spike
        #x = (1.0 - spikes) * x + spikes * self.reset_x
        #y = (1.0 - spikes) * y + spikes * self.reset_y

        return spikes.float(), x, y