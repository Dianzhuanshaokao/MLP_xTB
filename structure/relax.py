# relax.py
from ase import Atoms
from ase.io import read, write
from ase.optimize import BFGS
from mace.calculators import MACECalculator
import torch

MAX_STEPS = 200          
FMAT_TOL = 0.1           
INPUT_FILE = 'Layer.xyz'
OUTPUT_FILE = 'Layer_relaxed.xyz'
MACE_MODEL = '/home/netszx/models/2024-01-07-mace-128-L2_epoch-199.model'

atoms = read(INPUT_FILE)
mace_calc = MACECalculator(model_paths = MACE_MODEL, device="cuda" if torch.cuda.is_available() else "cpu", default_dtype="float64")
atoms.calc = mace_calc

opt = BFGS(atoms, logfile='relax.log', trajectory='relax.traj')
opt.run(fmax=FMAT_TOL)

write(OUTPUT_FILE, atoms)
print(f"\n Mission complished:  {OUTPUT_FILE}")