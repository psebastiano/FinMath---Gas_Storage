"""Strategy networks: N small feed-forward nets, day k uses net day_to_net[k]."""
from torch import Tensor, nn


def make_day_to_net(month_start: Tensor, sharing: str) -> Tensor:
    """(K,) long: index of the network used on each day ("day" | "month" | "shared")."""
    raise NotImplementedError


class PolicyNetworks(nn.Module):
    """Outputs in (0, 1); rescaled to the allowed range in the rollout."""

    def __init__(self, n_inputs: int, n_outputs: int, day_to_net: Tensor,
                 hidden: int = 16, n_hidden_layers: int = 1):
        super().__init__()
        raise NotImplementedError

    def forward(self, k: int, x: Tensor) -> Tensor:
        """x: (B, n_inputs) -> (B, n_outputs) in (0, 1)."""
        raise NotImplementedError
