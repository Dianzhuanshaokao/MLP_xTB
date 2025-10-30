#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from xtb.ase.calculator import XTB
from ase.calculators.dftb import Dftb
import numpy as np
import os
from ase.io import read, write
from ase.md.langevin import Langevin
from ase import units
from mace.calculators import MACECalculator
import pickle
from src.reaction_detector import analyze_reaction
from src.hybrid_calculator import HybridCalculator

atoms = read('structure/Constrainted_Layer.xyz')
atoms.pbc = True


MACE_MODEL = '/home/netszx/models/2024-01-07-mace-128-L2_epoch-199.model'
mace_calc = MACECalculator(model_paths = MACE_MODEL, device="cuda" if torch.cuda.is_available() else "cpu", default_dtype="float64")
xtb_calc = XTB(method='GFN1-xTB',accuracy=2.0)


os.makedirs('outputs', exist_ok=True)

ACTIVE_LEARNING_FILE = "active_learning_pool.pkl"
MAX_STEPS = 10000      # Maximum MD steps
CHECK_INTERVAL = 100   # Check for reactions every N steps
ERROR_THRESHOLD_ENERGY = 0.1   # eV
ERROR_THRESHOLD_FORCE = 0.1    # eV/Å
FINETUNE_EVERY_N_SAMPLES = 30  # Number of samples to trigger fine-tuning
if os.path.exists(ACTIVE_LEARNING_FILE):
    with open(ACTIVE_LEARNING_FILE, 'rb') as f:
        active_learning_pool = pickle.load(f)
    print(f"Loaded {len(active_learning_pool)} samples from active learning pool.")
else:
    active_learning_pool = []
    print("Initialized empty active learning pool.")


dyn = Langevin(
    atoms,
    timestep=1.0 * units.fs,
    temperature_K=300,
    friction=0.01 / units.fs,
    trajectory='outputs/adaptive_mlmd.traj'
)

hybrid_calc = HybridCalculator(mace_calc, xtb_calc, R_inner=2.5, R_outer=4.5)

print("Setting energy offset for hybrid calculator...")
hybrid_calc.set_energy_offset(atoms)

print("Starting adaptive ML-MD...")

atoms.calc = hybrid_calc

for step in range(MAX_STEPS):
    dyn.run(steps=1)
    
    if step % CHECK_INTERVAL == 0:
        is_reactive, reactive_indices = analyze_reaction(atoms, verbose=True)
        if is_reactive:
            hybrid_calc.calculate(atoms=atoms, reactive_indices=reactive_indices)
            print(f"\n Reaction detected at step {step}")
            # Compute energies and forces
            E_hybrid = hybrid_calc.results['energy']
            F_hybrid = hybrid_calc.results['forces']

            E_mlp = atoms.get_potential_energy()
            F_mlp = atoms.get_forces()
            
            dE = abs(E_hybrid- E_mlp)
            dF = np.mean(np.linalg.norm(F_hybrid - F_mlp, axis=1))
            print(f"dE = {dE:.4f} eV, dF = {dF:.4f} eV/Å")

            # Check if errors exceed thresholds
            if dE > ERROR_THRESHOLD_ENERGY or dF > ERROR_THRESHOLD_FORCE:
                sample = {
                    'atoms': atoms.copy(), 
                    'energy': E_hybrid, 
                    'forces': F_hybrid.copy(), 
                    'step': step,
                    'reactive_indices': reactive_indices
                    }
                active_learning_pool.append(sample)
                print("Added to active learning pool")

                with open(ACTIVE_LEARNING_FILE, 'wb') as f:
                    pickle.dump(active_learning_pool, f)

                # Check if we need to trigger fine-tuning
                if len(active_learning_pool) >= FINETUNE_EVERY_N_SAMPLES:
                    print("\n Triggering MACE fine-tuning...")
                    print("Please run Cell 6 to fine-tune the model.")
                    break
        else:
            print(f"No reaction detected at step {step}.")
print("Adaptive MD finished.")
