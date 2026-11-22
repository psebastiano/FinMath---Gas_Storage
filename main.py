"""Reproduce every figure and table of the report: python main.py"""
import torch

from gas_storage.config import DataConfig, ModelConfig, TrainConfig
from gas_storage.market import SeasonalOUSimulator, make_data
from gas.storage.policy import PolicyNetworks, make_day_to_net
from gas.storage.rollout import rollout
from gas.storage.storage import StorageSpec
from gas.storage.experiments import EXPERIMENTS
from gas.storage.train import train

def run_model(name, model_cfg, train_cfg, train_data, test_data, spec, simulator):
    """Build the networks, train them, evaluate on the test paths"""
    print(f"\n=== {name} ===")
    torch.manual_seed(train_cfg.seed)

    n_outputs = 2 if model_cfg.trade_forwards else 1
    day_to_net = make_day_to_net(train_data.month_start, model_cfg.net_sharing)
    policy = PolicyNetworks(len(model_cfg.features), n_outputs, day_to_net,
                            model_cfg.hidden, model_cfg.n_hidden_layers)

    history = train(policy, train_data, spec, model_cfg, train_cfg, simulator)

    # Evaluation step
    with torch.no_grad():
        res_train = rollout(policy, train_data, spec, model_cfg, history["price_scale"])
        res_test = rollout(policy, test_data, spec, model_cfg, history["price_scale"])

    for label, res in [("train", res_train), ("test", res_test)]:
        print(f" {label:5s} P&L (M EUR): mean {res.pnl.mean():.3f} std {res.pnl.std():.3f}")
    
    print(f" max |H_K| on test: {res_test.level[:, -1].abs().max():.2e}")

    return policy, history, res_train, res_test

def main() -> None:
    simulator = SeasonalOUSimulator()
    train_data, test_data = make_data(simulator, DataConfig())
    spec = StorageSpec.paper(K=train_data.n_days)
    train_cfg = TrainConfig()

    for name in EXPERIMENTS.items():
        run_model(name, ModelConfig.smod(), train_cfg, train_data, test_data, spec, simulator)

if __name__ == "__main__":
    main()
