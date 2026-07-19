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
        threshold=1.0,
    ):
        super().__init__()

        self.omega = nn.Parameter(
            torch.empty(n).uniform_(0.15, 0.35)
        )

        self.damping = nn.Parameter(
            torch.empty(n).uniform_(-0.08, -0.02)
        )

        self.threshold = threshold

    def forward(self, x, y, I, dt=1.0):
        """
        x : membrane state
        y : resonant state
        I : input current
        """

        # Ensure damping stays negative
        damping = -torch.abs(self.damping)

        # Damped oscillator dynamics
        dx = damping * x - self.omega * y + I
        dy = self.omega * x + damping * y

        # Euler integration
        x = x + dt * dx
        y = y + dt * dy

        # Spike generation
        spikes = SurrogateSpike.apply(
            x,
            self.threshold
        )

        #reset = spikes.detach()  
        #x = x - reset

        return spikes.float(), x, y