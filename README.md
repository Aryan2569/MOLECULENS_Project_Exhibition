# MOLECULENS

**An ML surrogate for molecular screening using Graph Neural Networks.**
DSN2098 Project Exhibition-I — VIT Bhopal, SCAI.

## What this project actually is

Before a chemical can become a medicine or a new material, someone has to
know its properties — is it toxic, does it dissolve, does it cross the
blood-brain barrier. Today that means real lab experiments or quantum
chemistry simulations (DFT) that take hours per molecule. So millions of
candidate molecules never get screened at all.

MOLECULENS predicts those properties directly from a molecule's structure,
in under a second. A molecule is naturally a graph (atoms = nodes, bonds =
edges), so we train a Graph Neural Network on public datasets where the real
answers are already known, and use it as a fast pre-filter: screen thousands
cheaply, then spend the expensive simulations only on the promising ones.

**What makes it not a rehashed tutorial notebook:**

1. **Multi-task** — one shared model predicts several properties at once,
   not one narrow model per property.
2. **Comparative** — GCN vs GAT vs MPNN, benchmarked against a classical
   Random Forest baseline, so the results prove something instead of
   asserting GNNs are better.
3. **Explainable** — attention weights highlight *which atoms* drove a
   prediction, so a chemist can sanity-check it instead of trusting a
   black box.

---

## Setup — Windows 11

Open **PowerShell** (not CMD) in this folder. In File Explorer you can
shift+right-click the folder and choose "Open PowerShell window here".

```powershell
# 1. Confirm Python is installed and on PATH
python --version        # should print 3.10-3.12

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it  (Windows syntax, not "source")
.\venv\Scripts\Activate.ps1

# 4. Install everything
pip install -r requirements.txt

# 5. Sanity check
python smoketest.py
```

If `smoketest.py` prints **"Pipeline is fully wired"**, your environment is
confirmed working end to end and you can move on.

### Windows gotchas you will probably hit

**PowerShell refuses to run the activate script** (`running scripts is
disabled on this system`). Fix once, per user:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Then re-run the activate command. This is normal Windows security, not a
broken install.

**`python` opens the Microsoft Store instead of running.** Windows ships a
stub. Install real Python from python.org and tick **"Add python.exe to
PATH"** during install, then reopen PowerShell.

**GPU check.** Run `nvidia-smi` in PowerShell — it should list your RTX 4060.
If the command isn't found, install the current NVIDIA driver from
nvidia.com. Everything here runs on CPU too, just slower; these datasets are
small enough that CPU training is survivable if the GPU fights you.

**PyTorch Geometric.** The `pip install` above pulls the pure-Python version,
which is all this project needs. If you later add layers requiring the
compiled extensions (`torch-scatter`, `torch-sparse`), those historically
need a C++ build toolchain on Windows — get prebuilt wheels from the PyG
site rather than compiling them yourself.

**Paths use backslashes** on Windows: `python data\download.py`, not
`data/download.py`. Forward slashes usually work in Python itself, but not
always in PowerShell.

---

## Project structure

```
moleculens/
├── data/
│   └── download.py       # fetches ESOL/Tox21/BBBP via PyG's MoleculeNet loader
├── src/
│   ├── featurize.py       # SMILES -> graph (RDKit parsing + graph construction)
│   ├── models.py           # GCN, GAT, MPNN — same interface, swap freely
│   ├── train.py             # training loop, works with any of the 3 models
│   ├── baseline_rf.py        # Random Forest on Morgan fingerprints (classical baseline)
│   └── api.py                 # FastAPI /predict endpoint
└── smoketest.py               # run this first
```

## Running it

```powershell
python data\download.py                                   # cache the datasets
cd src
python baseline_rf.py                                     # classical baseline
python train.py --dataset ESOL --model gcn --epochs 50    # first real training run
uvicorn api:app --reload                                  # live demo API
```

## Roadmap against your review dates

- **Review-I** — concept only, no code required. But having `smoketest.py`
  actually running makes the concept slide far more credible than a diagram
  with nothing behind it.
- **Review-II (15-17 Sept)** — needs "40-60% module implementation and demo."
  `train.py` running end to end on ESOL with GCN clears that bar.
- **Final Review (5-7 Oct)** — add GAT/MPNN runs on Tox21 + BBBP, the RF
  comparison table, the explainability visualization, the React frontend,
  demo video, and testing section.
- **Report due 09-10-2026.**

## Known gotchas already fixed in this scaffold

- `sklearn.metrics.mean_squared_error(..., squared=False)` was removed in
  recent scikit-learn — `baseline_rf.py` uses `root_mean_squared_error`.
- `AllChem.GetMorganFingerprintAsBitVect` is deprecated — replaced with
  `rdFingerprintGenerator.GetMorganGenerator`.
- Tox21 contains unparsable SMILES rows — `featurize.py` skips them and
  reports a count instead of crashing.
- Single-atom molecules have no bonds — `featurize.py` gives them a self-loop
  so message passing doesn't choke on an empty edge index.

## Watch out for (real risk in your results)

Tox21 is heavily class-imbalanced — most molecules aren't toxic. A naive
model can score high accuracy by predicting "safe" every time and be
completely useless. Use class weighting or focal loss, and report AUC rather
than raw accuracy.

Related: if your GCN, GAT, MPNN *and* Random Forest all land at ~100%,
that's a data-leakage red flag to investigate, not a result to celebrate.
One of the papers you were given reports exactly this on QM9 and openly
flags it as a limitation of a too-simple dataset — good thing to cite as
awareness in your own report.

## Core references (Literature Review slide)

1. Wu, Z. et al. (2018). *MoleculeNet: A Benchmark for Molecular Machine
   Learning.* Chemical Science, 9(2), 513-530.
2. Gilmer, J. et al. (2017). *Neural Message Passing for Quantum Chemistry.*
   ICML.
3. Yang, K. et al. (2019). *Analyzing Learned Molecular Representations for
   Property Prediction.* J. Chem. Inf. Model., 59(8), 3370-3388.
4. Velickovic, P. et al. (2018). *Graph Attention Networks.* ICLR.
5. Ying, R. et al. (2019). *GNNExplainer: Generating Explanations for Graph
   Neural Networks.* NeurIPS.
