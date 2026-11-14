# Team meeting: to-do draft

Project: reimplementation of Curin et al. (2021), *A deep learning model for gas storage optimization*.
Deadline: **report + code ZIP + slides on Moodle, Thu 11 Dec 2026, 23:59**. Presentation: **Mon 21 Dec, 16:15**. <- I asked to move this, waiting for new date. 

---

## 1. Decisions to take at the meeting

- [x] **Code workflow:** local uv venv + GitHub repo, notebook run on Colab, repo public at submission (see §2).
- [ ] **Code structure and function signatures:** go through `notes/code_structure.md` together.
- [ ] **Work split** (proposal in §4): who takes the simulator, who takes SMod?
- [ ] **Data format** between simulator and model (proposal in §3). This must be fixed first.
- [ ] **Price model** for the simulator: seasonal mean-reverting (OU on log-price) to start? Spikes later?
- [ ] **Benchmark:** intrinsic LP first (easy), LSMC later if time allows?
- [ ] **Training data:** fixed 900/100 paths like the paper.
- [ ] **Storage parameters:** use the paper's ℓ_k / u_k regimes and c ≈ 250,000 MWh (inferred from Fig. 2).
- [ ] **Git rules:** branch per feature + pull requests, or push to `main`? Who reviews?
- [ ] Weekly sync day/time.

## 2. Environment and code workflow

Teacher's environment = **Colab runtime 2026.01**: Python 3.12.12, NumPy 2.0.2, SciPy 1.16.3, Matplotlib 3.10.0, PyTorch 2.9.0+cu126.
Code must run start to finish, submitted as a ZIP of `.py` file(s) or a project.

Proposed plan:
1. **Local `uv` venv** matching the teacher's versions exactly; develop locally.
2. **Logic lives in `.py` files** in `src/gas_storage/`. Those diff and merge cleanly in git.
3. **A notebook in the repo** (`notebooks/run_all.ipynb`) imports from `src/` and runs the experiments and plots. Its first cell clones the repo when run on Colab and skips itself locally.
4. Once it works locally: open the notebook in **Colab (runtime 2026.01)** and check that it runs start to finish.
5. **Make the repo public** before submission, so the clone cell works for the teacher. (Colab doesn't keep cloned files: every run starts on a fresh machine.)
6. **Submit**: ZIP of the repo on Moodle (official, must run on its own) + Colab link in the README (convenience).

- Use **only** numpy / scipy / matplotlib / torch (no pandas or other packages the teacher didn't list).
- Code structure and signatures: see `notes/code_structure.md`.

Housekeeping:
- [ ] Pin `requirements.txt` to the teacher's versions; drop `pandas` and `jupyter`.
- [ ] Fix the deadlines in `README.md` (currently says 10 Dec and 14 Dec, should be **11 Dec** and **21 Dec**).
- [ ] Add a `main.py` (or `run_all.py`) that reproduces every figure and table in one go.
- [ ] Fix random seeds everywhere (reproducibility is graded).
- [ ] README: setup + how to run (on Colab and locally).

## 3. Data interface (proposal: agree on this first; full version in `notes/code_structure.md`)

```python
# simulator output, all float32 torch tensors, units: €/MWh
@dataclass
class MarketData:
    spot: Tensor         # (M, K)  day-ahead price S_k on each path
    fwd: Tensor          # (M, K)  front-month forward price F_k (SFMod only; unused in last month)
    month_start: Tensor  # (J+1,)  first day of each month n_0=0 < n_1 < ... , plus K at the end

def simulate(M: int, K: int = 351, seed: int = 0) -> MarketData: ...
```

```python
# storage contract, independent of the market, units: MWh and MWh/day
@dataclass
class StorageSpec:
    capacity: float      # c ≈ 250_000
    inj_max: Tensor      # (K,)  u_k ≥ 0   (2808 for k ≤ 200, else 408)
    wdr_max: Tensor      # (K,)  ℓ_k ≤ 0   (−600 for k ≤ 170, else −3072)
    kappa: float = 0.0   # proportional cost
    fixed_cost: float = 0.0  # C
```

To unblock the model side on day 1: the simulator owner first ships a **dummy `simulate()`** (e.g. a plain seasonal curve plus noise) with the right shapes, then improves it.

## 4. Work split (proposal)

| Track | Person A: market | Person B: model |
|---|---|---|
| 1 | Dummy `simulate()` with the agreed shapes | Storage environment: state update H_{k+1} = H_k + h_k, tightened bounds ℓ̃/ũ, empty-at-end rule |
| 2 | Real spot model: seasonal + mean-reverting, roughly calibrated to public data (EIA Henry Hub / TTF) | Policy networks (N distinct nets, sigmoid + rescaling), loss = −mean U(W), Adam training loop |
| 3 | Forward curve F_k consistent with the spot model; drift / no-drift switch | Evaluation: P&L histograms, storage-level plots (paper Figs. 3, 6) |
| 4 | Benchmark: intrinsic value (LP with `scipy.optimize.linprog`) | Extend to SFMod (§5) |
| 5 | LSMC benchmark (if time) | α sweep, risk-aversion r sweep |

Both: unit-test the constraints (no action outside [ℓ̃, ũ], H_K = 0 on every path).

## 5. Design for extension SMod → SFMod (from day 1)

- One **environment loop** whose state can hold more than H (later also the forward delivery commitments d^j).
- Policy output dimension = **number of actions** (1 for SMod, 2 for SFMod), not hard-coded.
- Inputs as a **feature vector** builder (k, H_k, S_k [, F_k]), so adding inputs = one line.
- Utility U and the risk aversion r passed as parameters.
- Sanity test: **SFMod with α = 0 must reproduce SMod.**

## 6. Timeline (proposal)

| Week | Goal |
|---|---|
| 11–17 Nov | Repo workflow, data interface, dummy simulator, environment + constraint tests |
| 18–24 Nov | SMod trains; real spot simulator; intrinsic benchmark; first figures |
| 25 Nov–1 Dec | SFMod + forward simulation; (LSMC) |
| 2–8 Dec | Experiments (α, r, drift vs no drift), final figures, write report + slides |
| 9–11 Dec | Buffer: run everything on a fresh Colab runtime, proofread, **submit** |
| 12–20 Dec | Rehearse presentation, prepare Q&A |

## 7. Report (online LaTeX) and slides

- [ ] Report skeleton (≥ 8 pages): introduction and storage problem · mathematical setup · deep hedging method · our simulator · benchmarks · results (SMod, SFMod) · discussion and limitations · conclusion.
- [ ] Slides: template copied to `slides/gas_storage_presentation.pptx`. Agree on an outline and on who presents which part.
- [ ] Every figure in report and slides must be generated by the submitted code.

## 8. Discussion points already identified (for the report)

- LSMC benchmark is spot-only, so the SFMod vs LSMC comparison is uneven.
- SFMod's profit may come from drift in the forward scenarios (ℙ vs ℚ), so test with and without drift.
- The paper's forced-withdrawal rule triggers after the fact; we cap the upper bound before each action (strict guarantee).
- Typo in the paper's training criterion: it minimizes U instead of −U.
- Capacity c is not given, so we infer c ≈ 250,000 MWh from Fig. 2.
- The network only learns in states it visits (an empty start), so no guarantees elsewhere.
- No public code or data (Axpo's scenarios are proprietary).
