"""Training loop (Adam on -mean U(W))."""
import torch

from .config import ModelConfig, TrainConfig
from .market import MarketData, MarketSimulator
from .objective import loss_fn
from .policy import PolicyNetworks
from .rollout import rollout
from .storage import StorageSpec

def train(policy: PolicyNetworks, train_data: MarketData, spec: StorageSpec,
          model_cfg: ModelConfig, train_cfg: TrainConfig,
          simulator: MarketSimulator | None = None) -> dict:
    """Returns the training history (loss per epoch)."""

    g = torch.Generator().manual_seed(train_cfg.seed)

    price_scale = float(train_data.spot.mean())

    optimizer = torch.optim.Adam(policy.parameters(), lr=train_cfg.lr)
    history = {"loss": [], "price_scale":price_scale}

    for epoch in range(train_cfg.epochs):
        if train_cfg.fresh_paths and simulator is not None:
            batch = simulator.simulate(train_cfg.batch_size, seed=train_cfg.seed + 1 + epoch)
        else:
            idx = torch.randint(0, train_data.n_paths, (train_cfg.batch_size,), generator=g)
            batch = train_data.subset(idx)

        result = rollout(policy, batch, spec, model_cfg, price_scale)
        loss = loss_fn(result.pnl, model_cfg.risk_aversion)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history["loss"].append(loss.item())
        if epoch % 50 == 0 or epoch == train_cfg.epochs - 1:
            print(f"epoch {epoch:4d}  loss {loss.item():.4f}")

    return history
