# test73: Unified crystal+glass SiO2 CG force field at matched N=1000

Extends test72 (which jointly trained on an N=8 crystal + N=1000 glass) by
replacing the N=8 crystal with a REAL, matched-size N=1000 crystal dataset
(5x5x5 tiling of the 8-atom primitive cell, real Vashishta NVT 300K MD --
see `make_crystal_1000_data.py`), so both training domains now share the
same atom count and comparable cell size. This removes the main confound
in test72's result (its glass->crystal-supercell transfer test showed the
joint model does NOT recognize large-scale crystalline order, plausibly
because it never saw a crystal at N=1000 during training).

## Why this exists

Everything up to test72 ran on a single laptop CPU. test73 is sized for a
GPU-equipped supercomputer (bigger network: 128 hidden / 4 layers by
default, 50k steps by default, configurable via environment variables) so
the joint model can be trained properly at the N=1000 scale on both sides.

## Setup

```bash
git clone <this repo>
cd test73
python make_crystal_1000_data.py   # needs lmp_serial (LAMMPS) + ASE; ~few
                                    # minutes, generates data_crystal_1000/
# data_glass_1000/ must be supplied separately (see "Data" below)
python train_and_export.py
```

Environment variables (all optional, CPU/sane defaults otherwise):
- `TEST73_DEVICE` (`cuda` or `cpu`, auto-detected if unset)
- `TEST73_HIDDEN_DIM` (default 128)
- `TEST73_N_LAYERS` (default 4)
- `TEST73_BATCH_SIZE` (default 32)
- `TEST73_N_STEPS` (default 50000)

## Data

- `data_crystal_1000/positions.npy` + `manifest.npz`: generated locally by
  `make_crystal_1000_data.py` (self-contained, only needs test69's 24-atom
  primitive cell file and a LAMMPS Vashishta potential).
- `data_glass_1000/positions.npy` + `manifest.npz`: copy from
  `../diffusion_for_multi_scale_molecular_dynamics/experiments/
  sio2-glass-si-only-egnn-fp/positions.npy` (+ manifest.npz) -- this is the
  real 5-replica Vashishta NVT 300K glass trajectory (DM2's
  sio2_3000_glass_100k_sample0.dat topology) used throughout test70-72.

## Planned validation (not yet run)

Once trained, test whether a CG force field trained ONLY on equilibrium
crystal and glass structures (never on any nucleation/transition pathway)
can nonetheless show glass -> crystal nucleation during long CGMD -- i.e.,
does the learned energy landscape have the crystal as a deeper basin that
a glass-started trajectory can find via thermal fluctuation + the model's
own dynamics, the same way real SiO2 can (slowly) crystallize under the
right conditions. This requires much longer MD (likely >> 2000 steps) than
anything run in test70-72, and the outcome is genuinely uncertain -- CG
force fields derived this way have no guarantee of getting barrier heights
right even if the two endpoint basins are individually stable.
