# relax.py
from ase import Atoms
from ase.io import read, write
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from mace.calculators import MACECalculator
import torch
import os

MAX_STEPS = 200          
FMAT_TOL = 0.1           
INPUT_FILE = os.path.join('Constrainted_Layer.xyz')
OUTPUT_FILE = os.path.join('Constrainted_Layer_relaxed.xyz')
MACE_MODEL = '/home/netszx/models/2024-01-07-mace-128-L2_epoch-199.model'


atoms = read(INPUT_FILE)
# Determine which atoms to fix: z-coordinate less than cutoff
z_positions = atoms.positions[:, 2]  # 所有原子的 z 坐标
cutoff_z = 20.8
fixed_mask = z_positions > cutoff_z
constraint = FixAtoms(mask=fixed_mask)
atoms.set_constraint(constraint)

print(f"Fixed {fixed_mask.sum()} atoms (z > {cutoff_z:.3f})")

mace_calc = MACECalculator(model_paths = MACE_MODEL, device="cuda" if torch.cuda.is_available() else "cpu", default_dtype="float64")
atoms.calc = mace_calc

opt = BFGS(atoms, logfile=os.path.join('relax.log'), trajectory=os.path.join('relax.traj'))
opt.run(fmax=FMAT_TOL)

write(OUTPUT_FILE, atoms)
print(f"\n Mission complished:  {OUTPUT_FILE}")