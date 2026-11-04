# FinMath – Gas Storage

Seminar project for **Machine Learning in Financial Mathematics** (LMU Munich, WS 2026/27, Zhipeng Huang).

We study and reimplement in PyTorch:

> N. Curin, M. Kettler, X. Kleisinger-Yu, V. Komaric, T. Krabichler, J. Teichmann, H. Wutte.
> *A deep learning model for gas storage optimization.* arXiv:2102.01980 (2021).

The paper optimizes the operation plan of an underground natural gas storage facility with a
deep-hedging-style neural policy and compares it to a least-squares Monte-Carlo (LSMC) baseline.

## Deliverables

| What | Deadline |
|---|---|
| Group + paper confirmation | 31 Oct 2026 |
| Report (≥ 8 pages) + code | 10 Dec 2026 |
| Slides | before the presentation (from 14 Dec 2026) |

## Layout

```
src/gas_storage/   model, market simulation, training, LSMC baseline
notebooks/         experiments and figures
report/            LaTeX report
slides/            presentation
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (source .venv/bin/activate on Linux)
pip install -r requirements.txt
```
