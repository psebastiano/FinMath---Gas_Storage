import math
import torch

from .market import MarketData

MONTH_START = torch.tensor([0, 30, 61, 91, 122, 153, 183, 214, 244, 275, 306, 334, 351])

def mock_market_data(n_paths: int, seed: int=0) -> MarketData:
    K = int(MONTH_START[-1])
    g = torch.Generator().manual_seed(seed)

    k = torch.arange(K, dtype=torch.float32)
    mean = 20.0 + 5.0 * torch.cos(2 * math.pi * (k - 290) / 365)

    # Spot = mean + independent noise per path and per day
    spot = mean + 1.0 * torch.randn(n_paths, K, generator=g)              # (M, K)

    # Forward price
    fwd = torch.full((n_paths, K), float("nan"))
    n_months = len(MONTH_START) - 1
    
    for j in range(n_months - 1):
        start, end = int(MONTH_START[j]), int(MONTH_START[j + 1])
        next_end  = int(MONTH_START[j + 2])
        next_month_mean = mean[end:next_end].mean()
        fwd[:, start:end] = next_month_mean + 0.5 * torch.randn(n_paths, end - start, generator = g)

    return MarketData(spot=spot, fwd=fwd, month_start = MONTH_START)


if __name__ == "__main__":
    # Quick visual check: python -m gas_storage.mock_market
    import matplotlib.pyplot as plt

    data = mock_market_data(n_paths=1000, seed=0)
    print("spot", tuple(data.spot.shape), "fwd", tuple(data.fwd.shape))

    fig, ax = plt.subplots(figsize=(10, 4))
    for i in range(3):
        ax.plot(data.spot[i], lw=0.7, alpha=0.6, label=f"spot, path {i}")
    ax.plot(data.spot.mean(0), color="black", lw=2, label="spot, mean over paths")
    ax.plot(data.fwd[0], color="tab:red", lw=1.5, label="front-month forward, path 0")
    for start in MONTH_START:
        ax.axvline(start, color="grey", lw=0.5, ls=":")
    ax.set_xlabel("day k")
    ax.set_ylabel("price (EUR/MWh)")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.show()
