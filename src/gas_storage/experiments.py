from .config import ModelConfig

EXPERIMENTS = {
    "smod":       ModelConfig.smod(),
    "sfmod_a0.1": ModelConfig.sfmod(alpha=0.1),
    "sfmod_a0.5": ModelConfig.sfmod(alpha=0.5),
}
