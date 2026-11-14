"""Training loop (Adam on -mean U(W))."""
from .config import ModelConfig, TrainConfig
from .market import MarketData, MarketSimulator
from .policy import PolicyNetworks
from .storage import StorageSpec


def train(policy: PolicyNetworks, train_data: MarketData, spec: StorageSpec,
          model_cfg: ModelConfig, train_cfg: TrainConfig,
          simulator: MarketSimulator | None = None) -> dict:
    """Returns the training history (loss per epoch)."""
    raise NotImplementedError
