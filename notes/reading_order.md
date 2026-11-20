# Reading order of `src/gas_storage/`

The order follows the data flow. My files come first, Gabriel's last.

| # | File | Contents | Owner |
|---|---|---|---|
| 1 | [config.py](../src/gas_storage/config.py) | `ModelConfig` (SMod/SFMod switches: α, cap, features, network size, r) and `TrainConfig` | me |
| 2 | [market.py](../src/gas_storage/market.py) | `MarketData`: the container I *receive*, holding `spot`, `fwd` and `month_start` | Gabriel (read it for the interface) |
| 3 | [storage.py](../src/gas_storage/storage.py) | `StorageSpec` (c, ℓ, u) and `flow_bounds`, i.e. eq. (17) | me |
| 4 | [policy.py](../src/gas_storage/policy.py) | the N networks and the day→network map | me |
| 5 | [rollout.py](../src/gas_storage/rollout.py) | **the core**: day-by-day loop with features → network → bounded flow → H update → P&L | me |
| 6 | [objective.py](../src/gas_storage/objective.py) | exponential utility and the loss | me |
| 7 | [train.py](../src/gas_storage/train.py) | the Adam loop | me |
| 8 | [evaluate.py](../src/gas_storage/evaluate.py) | stats and plots | shared |
| 9 | [benchmarks.py](../src/gas_storage/benchmarks.py) | LSMC, intrinsic value, perfect foresight | Gabriel |

[code_structure.md](code_structure.md) explains the reasoning behind each signature, so keep it open next to the code.

## Mock market data

To test the rollout without waiting for Gabriel, I put a mock in its own file, `src/gas_storage/mock_market.py`, so that it never conflicts with his `market.py`. The mock returns a `MarketData` with:

- **spot:** one seasonal curve shared by all paths, plus random noise per path, e.g. `S_k = 20 + 5·cos(2π·k/365) + noise_k`, with the phase set so that winter is expensive. Don't give each path its own sine: each path would then become predictable after a few days.
- **fwd:** the seasonal curve averaged over next month, plus a bit of noise, so that F stays roughly consistent with S.
- **shapes:** `spot` (M, K), `fwd` (M, K), `month_start`.

Only the script that creates the data picks the source. Everything downstream takes a `MarketData` and doesn't care where it came from.
