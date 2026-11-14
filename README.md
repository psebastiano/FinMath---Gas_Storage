# FinMath – Gas Storage

Seminar project for **Machine Learning in Financial Mathematics** (LMU Munich, WS 2026/27, Zhipeng Huang).

We study and reimplement in PyTorch:

> N. Curin, M. Kettler, X. Kleisinger-Yu, V. Komaric, T. Krabichler, J. Teichmann, H. Wutte.
> *A deep learning model for gas storage optimization.* Decisions in Economics and Finance 44 (2021), arXiv:2102.01980.

The paper optimizes the operation plan of an underground natural gas storage facility with a
deep-hedging-style neural policy and compares it to a least-squares Monte-Carlo (LSMC) baseline.

## Deliverables

| What | Deadline |
|---|---|
| Group + paper confirmation | 31 Oct 2026 |
| Report (≥ 8 pages) + code + slides (Moodle) | 11 Dec 2026, 23:59 |
| Presentation | 21 Dec 2026, 16:15 |

## Layout

```
src/gas_storage/   simulator, storage contract, policy networks, rollout, training, benchmarks
notebooks/         run_all.ipynb: runs the experiments (locally or on Colab)
main.py            reproduces every figure and table of the report
report/            LaTeX report
slides/            presentation
notes/             design notes
```

## Environment

Required versions (= Colab runtime 2026.01):
Python 3.12.12, NumPy 2.0.2, SciPy 1.16.3, Matplotlib 3.10.0, PyTorch 2.9.0+cu126.

### On Colab
Open `notebooks/run_all.ipynb` in Colab, select runtime version **2026.01**, and run all cells.
The first cell clones this repository and installs the package.

### Locally (with [uv](https://docs.astral.sh/uv/))
```bash
uv python install 3.12.12
uv venv --python 3.12.12 .venv
uv pip install --python .venv -r requirements.txt --index-strategy unsafe-best-match
uv pip install --python .venv -e .
```
Then activate (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on Linux/macOS) and run `python main.py`.

For development (Jupyter kernel for the notebook): install `requirements-dev.txt` instead of `requirements.txt`.
