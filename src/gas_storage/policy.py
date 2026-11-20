"""Strategy networks: N small feed-forward nets, day k uses net day_to_net[k]."""
import torch
from torch import Tensor, nn


def make_day_to_net(month_start: Tensor, sharing: str) -> Tensor:
    """(K,) long: index of the network used on each day ("day" | "month" | "shared")."""
    
    K = int(month_start[-1]) #The last element of month start is the last day. I convert it to an int to use it as a bound.
                            
    #Init the lookup table with 0
    day_to_net = torch.zeros(K, dtype=torch.long)

    if sharing == "day":
        for i in range(K):
            day_to_net[i] = i
    elif sharing == "month":
        for i in range(K):
            for j in range(len(month_start)):
                if i < month_start[j]:
                    day_to_net[i] = j-1
                    break
    elif sharing == "shared":
        pass #All zeros : so each day get one net
    else:
        raise ValueError(f"Unknown sharing : {sharing}")

    return day_to_net


class PolicyNetworks(nn.Module):
    """Outputs in (0, 1); rescaled to the allowed range in the rollout."""

    def __init__(self, n_inputs: int, n_outputs: int, day_to_net: Tensor,
                 hidden: int = 16, n_hidden_layers: int = 1):
        super().__init__()

        #Compute the needed number of nets
        N = torch.unique(day_to_net).numel()

        self.register_buffer("day_to_net", day_to_net)

        self.nets = nn.ModuleList(
            [self._make_net(n_inputs, n_outputs, hidden, n_hidden_layers) for _ in range(N)]
        )

    @staticmethod
    def _make_net(n_inputs: int, n_outputs: int, hidden: int, n_hidden_layers: int) -> nn.Sequential:
        """One feed-forward net: n_hidden_layers sigmoid layers of width hidden, then a sigmoid output."""
        layers = []
        width = n_inputs
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(width, hidden))
            layers.append(nn.Sigmoid())
            width = hidden
        layers.append(nn.Linear(width, n_outputs))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)


    def forward(self, k: int, x: Tensor) -> Tensor:
        """x: (B, n_inputs) -> (B, n_outputs) in (0, 1)."""
        net = self.nets[int(self.day_to_net[k])]
        return net(x)
