 

```python
"""Return n_paths simulated price paths, reproducible with seed."""
      
```

# Proposed code structure (to agree on at the meeting)

Goal: one code path for both models. **SMod = SFMod with forward trading switched off.**
Everything is vectorized over paths (batch dimension B) and loops over the K days.

```
src/gas_storage/
  config.py      # all dataclass configs (ModelConfig, TrainConfig)
  market.py      # MarketData + simulators            ← person A
  storage.py     # StorageSpec + bound computations   ← person B
  policy.py      # PolicyNetworks (the N small nets)   ← person B
  rollout.py     # runs a strategy through the year    ← person B
  objective.py   # utility + loss                      ← person B
  train.py       # training loop                       ← person B
  benchmarks.py  # intrinsic LP, perfect foresight, LSMC  ← person A
  evaluate.py    # statistics + plots (paper Figs. 3, 5, 6)  ← both
main.py          # reproduces every figure/table of the report
notebooks/run_all.ipynb   # thin: clone (Colab) → import → run
tests/           # constraint and consistency tests
```

---

## 0. Conventions (agree first)

| Item           | Proposal                                                                                              |
| -------------- | ----------------------------------------------------------------------------------------------------- |
| Units          | prices €/MWh, volumes MWh, rates MWh/day                                                             |
| Tensors        | `torch.float32`, shape `(M, K)` = (paths, days)                                                   |
| Horizon        | K = 351 days, J = 12 months                                                                           |
| Month calendar | `month_start = [n_0=0, n_1, …, n_{J-1}, n_J=K]`, about 29 days per month                           |
| P&L scale      | P&L reported in**millions** (`pnl_scale = 1e6`), so that r = 3 is meaningful, as in the paper |
| Sign           | h > 0 = buy/inject, h < 0 = sell/withdraw; W = Σ −h·price                                          |
| Seeds          | every random function takes a`seed` argument                                                        |

---

## 1. `market.py` (person A)

```python
@dataclass
class MarketData:
    spot: Tensor                 # (M, K)  day-ahead price S_k
    fwd: Tensor | None           # (M, K)  price on day k of the contract delivering NEXT month
                                 #         (front month); NaN during the last month
    month_start: Tensor          # (J+1,) long, n_0 = 0 < ... < n_J = K

    @property
    def n_paths(self) -> int: ...
    @property
    def n_days(self) -> int: ...
    def month_of_day(self) -> Tensor: ...          # (K,) long, month index of each day
    def subset(self, idx: Tensor) -> "MarketData": ...   # select paths (batches, train/test)
    def to(self, device) -> "MarketData": ...

def train_test_split(data: MarketData, n_train: int, seed: int) -> tuple[MarketData, MarketData]: ...
```

```python
class MarketSimulator(Protocol):
    def simulate(self, n_paths: int, seed: int) -> MarketData:
	"""Return n_paths simulated price paths, reproducible with seed."""
      
class DummySimulator:                 # day 1: right shapes, simple seasonal curve + noise
    def __init__(self, K: int = 351, n_months: int = 12): ...
    def simulate(self, n_paths: int, seed: int) -> MarketData: ...

@dataclass
class SeasonalOUConfig:
    K: int = 351
    n_months: int = 12
    mean_level: float = 20.0          # €/MWh, average seasonal level
    seasonal_amp: float = 5.0         # winter–summer amplitude
    mean_reversion: float = 0.05      # per day
    vol: float = 0.03                 # daily volatility of log-price
    fwd_premium: float = 0.0          # drift in forwards (0 = martingale, no free forward profit)

class SeasonalOUSimulator:            # log S = seasonal mean + Ornstein–Uhlenbeck deviation
    def __init__(self, cfg: SeasonalOUConfig): ...
    def seasonal_mean(self) -> Tensor: ...         # (K,)
    def simulate(self, n_paths: int, seed: int) -> MarketData: ...
```

Forward price idea: F_k = average over next month's days of E[S_t | S_k] (closed form for OU), plus an optional premium. With `fwd_premium = 0` the forward has no drift, which gives the "with vs without drift" experiment for free.

---

## 2. `storage.py` (person B)

```python
@dataclass
class StorageSpec:
    capacity: float                  # c (MWh)
    inj_max: Tensor                  # (K,)  u_k ≥ 0
    wdr_max: Tensor                  # (K,)  ℓ_k ≤ 0
    kappa: float = 0.0               # proportional cost
    fixed_cost: float = 0.0          # C

    @staticmethod
    def paper(K: int = 351, capacity: float = 250_000.0) -> "StorageSpec": ...
        # u = 2808 (k ≤ 200) else 408 ;  ℓ = −600 (k ≤ 170) else −3072

    def remaining_withdrawal(self) -> Tensor: ...  # (K+1,)  R_k = Σ_{j ≥ k} |ℓ_j|, R_K = 0

def flow_bounds(spec: StorageSpec, k: int, level: Tensor) -> tuple[Tensor, Tensor]:
    """Bounds on the PHYSICAL flow q_k into storage on day k, shape (B,).
    lo = max(ℓ_k, −H_k)
    hi = min(u_k, c − H_k, R_{k+1} − H_k)    # last term: strict 'empty at the end' guarantee
    """
```

---

## 3. The key idea: physical flow vs trades (this makes SMod ⊂ SFMod)

On every day k in month m:

| Quantity | Meaning                                   | SMod    | SFMod                                   |
| -------- | ----------------------------------------- | ------- | --------------------------------------- |
| q_k      | physical flow into storage, in [lo, hi]   | = h^S_k | = h^S_k + d^m                           |
| d^m      | daily forward delivery this month (fixed) | 0       | Σ of forward trades made in month m−1 |
| h^S_k    | spot trade                                | q_k     | **q_k − d^m**                    |
| h^F_k    | forward trade (front month)               | 0       | network output, capped by α            |

So the network always chooses **q_k** (where the storage goes), and the spot trade is what is left after the delivery. The storage update is the same in both models: H_{k+1} = H_k + q_k.
With `trade_forwards=False` (or α = 0) every d^m = 0, so the code reduces exactly to SMod.

P&L (in millions):

W = [ Σ_k −h^S_k·S_k  +  Σ_k −h^F_k·F_k·(days in next month)  −  κ·Σ|h·price|  −  C ] / pnl_scale

---

## 4. `config.py`

```python
@dataclass
class ModelConfig:
    trade_forwards: bool = False         # False = SMod, True = SFMod
    alpha: float = 0.0                   # forward volume cap (eq. 16)
    fwd_cap: str = "per_trade"           # "per_trade" (paper literal) | "cumulative" (see open questions)
    features: tuple[str, ...] = ("k", "H", "S")   # SFMod: ("k", "H", "S", "F", "D")
    net_sharing: str = "month"           # "day" (N=K) | "month" (N=12) | "shared" (N=1)
    hidden: int = 16
    n_hidden_layers: int = 1             # L = 2 in the paper's notation
    risk_aversion: float = 3.0           # r
    pnl_scale: float = 1e6

    @staticmethod
    def smod() -> "ModelConfig": ...
    @staticmethod
    def sfmod(alpha: float) -> "ModelConfig": ...

@dataclass
class TrainConfig:
    epochs: int = 1000
    batch_size: int = 64
    lr: float = 0.05
    fresh_paths: bool = False            # True: new simulated paths every batch
    seed: int = 0
    device: str = "cpu"
```

---

## 5. `policy.py` (person B)

```python
def make_day_to_net(month_start: Tensor, sharing: str) -> Tensor: ...   # (K,) long, day → net index

class PolicyNetworks(nn.Module):
    """N small feed-forward nets; day k uses net day_to_net[k]. Outputs in (0, 1)."""
    def __init__(self, n_inputs: int, n_outputs: int, day_to_net: Tensor,
                 hidden: int = 16, n_hidden_layers: int = 1): ...
    def forward(self, k: int, x: Tensor) -> Tensor: ...    # x: (B, n_inputs) → (B, n_outputs) ∈ (0,1)
```

`n_outputs = 2 if cfg.trade_forwards else 1`.

---

## 6. `rollout.py` (person B): the heart, shared by SMod and SFMod

```python
def build_features(k: int, level: Tensor, market: MarketData, fwd_pos: Tensor,
                   spec: StorageSpec, cfg: ModelConfig) -> Tensor: ...
    # normalized inputs, only day-k information (no peeking!): k/K, H/c, S_k/scale, F_k/scale, D/cap

@dataclass
class RolloutResult:
    flow: Tensor        # (B, K)   q_k
    h_spot: Tensor      # (B, K)
    h_fwd: Tensor       # (B, K)   zeros for SMod
    level: Tensor       # (B, K+1) H_0 … H_K
    pnl: Tensor         # (B,)     W (scaled)
    pnl_spot: Tensor    # (B,)
    pnl_fwd: Tensor     # (B,)

def rollout(policy: PolicyNetworks, market: MarketData, spec: StorageSpec,
            cfg: ModelConfig) -> RolloutResult: ...
```

Inside the loop over k:

1. `lo, hi = flow_bounds(spec, k, H)`
2. `y = policy(k, build_features(...))`
3. `q = lo + y[:, 0] * (hi - lo)`
4. if forwards and not last month: `h_F = a_lo + y[:, 1] * (a_hi - a_lo)` (α cap), accumulate the position for next month
5. `h_S = q - d[current month]`
6. `H = H + q`, accumulate the P&L

Gradients flow through everything (no `.detach()` on H).

---

## 7. `objective.py` and `train.py`

```python
def exp_utility(w: Tensor, r: float) -> Tensor: ...        # (1 − e^{−r w}) / r
def loss_fn(pnl: Tensor, r: float) -> Tensor: ...          # −mean U(W)   (note the minus: paper typo)

def train(policy: PolicyNetworks, train_data: MarketData, spec: StorageSpec,
          model_cfg: ModelConfig, train_cfg: TrainConfig,
          simulator: MarketSimulator | None = None) -> dict:   # history: loss per epoch
    ...
```

---

## 8. `benchmarks.py` (person A)

```python
def intrinsic_lp(price_curve: np.ndarray, spec: StorageSpec) -> tuple[np.ndarray, float]: ...
    # deterministic plan on the expected price curve (scipy.optimize.linprog) → lower-bound benchmark
def perfect_foresight(market: MarketData, spec: StorageSpec) -> np.ndarray: ...
    # LP per path knowing the future → UPPER bound (cheats, only for reference)
def lsmc(market: MarketData, spec: StorageSpec, n_levels: int = 101,
         basis_degree: int = 3) -> np.ndarray: ...
    # the paper's benchmark; returns the P&L per path (if time allows)
```

Any good strategy should land between intrinsic and perfect foresight.

---

## 9. `evaluate.py` (both)

```python
def summary_stats(pnl: Tensor) -> dict: ...        # mean, median, std (paper's Fig. 6 table)
def plot_paths(market, ax=None): ...               # Fig. 3 top-left
def plot_levels(result, spec, ax=None): ...        # Fig. 3 top-right
def plot_pnl_hist(pnls: dict[str, Tensor], ax=None): ...   # Figs. 3 / 5
def plot_pnl_box(pnls: dict[str, Tensor], ax=None): ...    # Fig. 6
```

---

## 10. `tests/`

- shapes: simulator output matches `MarketData` conventions
- constraints: with a **random** policy, 0 ≤ H ≤ c, flows within [ℓ, u], **H_K = 0** on every path
- consistency: **SFMod with α = 0 gives the same result as SMod** (same seed)
- no peeking: features at day k don't change when prices after day k are modified

---

## Open questions for the meeting

1. **Forward cap (eq. 16):** the paper caps each **daily trade** at a = α·c/(days in delivery month), which keeps one day's trade storable (total delivery ≤ α·c). But the agent trades on ~29 days, so the position can add up to ~29·a: a monthly delivery of ~29·α·c (≈ 15·c for α = 0.5), far more than the storage holds. Proposal: cap the **cumulative** position so that a month's total delivery ≤ α·c. Keep the literal version as an option.
2. **SFMod inputs:** the paper lists (k, H, S, F). We propose adding the current forward position D (otherwise the net can't know what it already bought). Report this as a deviation.
3. **Sell side of forwards:** the paper only gives an upper bound; we use a symmetric cap [−a, a].
4. **Month calendar:** how to split 351 days into 12 months (equal ~29-day months?).
5. **Simulator parameters:** calibrate to which public data (EIA Henry Hub vs TTF)?
