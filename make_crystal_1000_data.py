"""Generate a real, thermally-equilibrated 1000-Si (3000-atom) SiO2
CRYSTAL dataset for test73, by tiling test69's 24-atom (8Si+16O) primitive
cell 5x5x5 (-> 3000 atoms, 1000 Si) and running real Vashishta NVT 300K MD
on it -- the crystal counterpart to the existing 1000-Si glass dataset
(toy-model/SiO2-CG/diffusion_for_multi_scale_molecular_dynamics/experiments/
sio2-glass-si-only-egnn-fp/positions.npy), so test73 can train a single
model on two REAL, matched-size (N=1000 CG beads) datasets: one ordered
(crystal), one disordered (glass).
"""
import subprocess
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read
from ase.io.lammpsdata import write_lammps_data

TEST69_DATA = Path("/Users/harutokono/ScoreMD/toy-model/SiO2-CG/test69/silica_primitive_init.data")
WORK_DIR = Path(__file__).resolve().parent / "crystal_1000_work"
POTENTIALS_DIR = Path(
    "/Users/harutokono/ScoreMD/toy-model/SiO2-CG/diffusion_for_multi_scale_molecular_dynamics"
    "/.venv/lib/python3.10/site-packages/lammps/share/lammps/potentials"
)
TILE = 5
TEMP_K = 300.0
EQ_STEPS = 20000
PROD_STEPS = 100000
DUMP_EVERY = 200  # -> 500 frames


def main():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    atoms = read(str(TEST69_DATA), format="lammps-data", style="atomic")
    # test69 convention: type 1 = Si, type 2 = O (see make_primitive_data.py)
    symbols = ["Si" if n == 1 else "O" for n in atoms.get_atomic_numbers()]
    atoms.set_chemical_symbols(symbols)
    big = atoms.repeat((TILE, TILE, TILE))
    print(f"Tiled {TILE}x{TILE}x{TILE}: {len(big)} atoms "
          f"({sum(1 for s in big.get_chemical_symbols() if s == 'Si')} Si)")

    data_path = WORK_DIR / "crystal_1000_init.data"
    with open(data_path, "w") as f:
        write_lammps_data(f, big, specorder=["Si", "O"], units="metal", atom_style="atomic")

    in_path = WORK_DIR / "nvt.in"
    dump_path = WORK_DIR / "crystal_1000_traj.lammpstrj"
    in_path.write_text(f"""
units metal
boundary p p p
atom_style atomic
read_data {data_path.name}
mass 1 28.0855
mass 2 15.999
pair_style vashishta
pair_coeff * * {POTENTIALS_DIR}/SiO.1990.vashishta Si O
velocity all create {TEMP_K} 42 mom yes rot yes dist gaussian
fix eqnvt all nvt temp {TEMP_K} {TEMP_K} 0.1
run {EQ_STEPS}
unfix eqnvt
fix prodnvt all nvt temp {TEMP_K} {TEMP_K} 0.1
dump traj all custom {DUMP_EVERY} {dump_path.name} id type xu yu zu
dump_modify traj sort id append no format float %.6f
run {PROD_STEPS}
""")
    result = subprocess.run(["lmp_serial", "-in", str(in_path)], cwd=str(WORK_DIR),
                             capture_output=True, text=True, timeout=1800)
    print(result.stdout[-2000:])
    if result.returncode != 0:
        print("LAMMPS FAILED:", result.stderr[-2000:])
        return

    text = dump_path.read_text()
    blocks = text.split("ITEM: TIMESTEP")[1:]
    frames = []
    cell = None
    for block in blocks:
        lines = block.strip().splitlines()
        n_atoms = int(lines[lines.index("ITEM: NUMBER OF ATOMS") + 1])
        bounds_idx = next(i for i, line in enumerate(lines) if line.startswith("ITEM: BOX BOUNDS"))
        bounds = [lines[bounds_idx + 1 + i].split() for i in range(3)]
        lengths = np.array([float(hi) - float(lo) for lo, hi in bounds])
        if cell is None:
            cell = lengths
        atoms_idx = next(i for i, line in enumerate(lines) if line.startswith("ITEM: ATOMS"))
        header = lines[atoms_idx].split()[2:]
        id_col, type_col = header.index("id"), header.index("type")
        x_col, y_col, z_col = header.index("xu"), header.index("yu"), header.index("zu")
        rows = [lines[atoms_idx + 1 + i].split() for i in range(n_atoms)]
        rows.sort(key=lambda r: int(r[id_col]))
        types = np.array([int(r[type_col]) for r in rows])
        pos = np.array([[float(r[x_col]), float(r[y_col]), float(r[z_col])] for r in rows])
        frames.append(pos[types == 1])  # Si only (specorder Si=1,O=2)

    si_frames = np.stack(frames, axis=0)
    print(f"Extracted {si_frames.shape[0]} frames, {si_frames.shape[1]} Si atoms")
    out_dir = Path(__file__).resolve().parent / "data_crystal_1000"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "positions.npy", (si_frames / 10.0).astype(np.float32))  # Angstrom -> nm
    np.savez(out_dir / "manifest.npz", cell_nm=(cell / 10.0).astype(np.float32),
              source=f"{TILE}x{TILE}x{TILE} tiled beta-cristobalite primitive cell, Vashishta NVT 300K")
    print(f"Wrote {out_dir}/positions.npy, cell_nm={cell / 10.0}")


if __name__ == "__main__":
    main()
