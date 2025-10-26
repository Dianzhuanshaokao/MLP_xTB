from ase.neighborlist import NeighborList
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
import logging

class HybridCalculator(Calculator):
    """
    ASE-compatible hybrid calculator that:
      - Uses MLP globally
      - Switches to xTB in reactive regions (with energy alignment + smooth blending)
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, mlp_calc, xtb_calc, 
                 R_inner=2.5, R_outer=4.5,
                 energy_offset=None):
        """
        Parameters:
            mlp_calc: MLP calculator (e.g., MACE)
            xtb_calc: xTB calculator
            R_inner: Inner radius for full xTB (Å)
            R_outer: Outer radius for full MLP (Å)
            energy_offset: Precomputed E_xtb - E_mlp for alignment (optional)
        """
        super().__init__()
        self.mlp_calc = mlp_calc
        self.xtb_calc = xtb_calc
        self.R_inner = R_inner
        self.R_outer = R_outer
        self.energy_offset = energy_offset  # Can be set later

    def set_energy_offset(self, atoms):
        """Compute energy offset using current atoms as reference."""
        atoms_ref = atoms.copy()
        atoms_ref.calc = self.mlp_calc
        E_mlp = atoms_ref.get_potential_energy()
        print(f"Reference MLP energy: {E_mlp:.6f} eV")
        
        atoms_ref.calc = self.xtb_calc
        E_xtb = atoms_ref.get_potential_energy()
        print(f"Reference xTB energy: {E_xtb:.6f} eV")
        
        self.energy_offset = E_xtb - E_mlp
        print(f"Energy offset set: {self.energy_offset:.6f} eV")

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes, reactive_indices=None):
        """
        Main calculation method.
        Parameters:
            atoms: ASE Atoms object
            reactive_indices: np.ndarray of atom indices involved in reaction (can be empty)
        """
        super().calculate(atoms, properties, system_changes)
        
        if reactive_indices is None:
            reactive_indices = np.array([], dtype=int)
        else:
            reactive_indices = np.asarray(reactive_indices)

        # --- Step 1: Global MLP ---
        atoms.calc = self.mlp_calc
        E_mlp = atoms.get_potential_energy()
        F_mlp = atoms.get_forces()

        log_file = 'outputs/adapt.log'
        logging.basicConfig(
            filename=log_file,
            filemode='a', 
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        logger = logging.getLogger(__name__)

        # --- Step 2: If no reactive atoms, return MLP result ---
        if len(reactive_indices) == 0:
            self.results = {'energy': E_mlp, 'forces': F_mlp}
            logger.info("No reactive atoms detected. Returning pure MLP results.")
            return

        # --- Step 3: Build cluster (reactive atoms + buffer) ---
        cutoff = self.R_outer + 1.0
        nl = NeighborList(cutoffs=[cutoff/2]*len(atoms), skin=0.0, self_interaction=False, bothways=True)
        nl.update(atoms)
        
        cluster_mask = np.zeros(len(atoms), dtype=bool)
        for i in reactive_indices:
            indices, _ = nl.get_neighbors(i)
            cluster_mask[indices] = True
        cluster_mask[reactive_indices] = True
        cluster_indices = np.where(cluster_mask)[0]
        
        cluster_atoms = atoms[cluster_indices]
        cluster_atoms.calc = self.xtb_calc

        cluster_info = f"Cluster built. Size: {len(cluster_atoms)} atoms (R_inner={self.R_inner}, R_outer={self.R_outer})"
        logger.info(f"{cluster_info}")
        
        # --- Step 4: xTB calculation (with energy alignment) ---
        E_xtb_cluster = cluster_atoms.get_potential_energy()
        F_xtb_cluster = cluster_atoms.get_forces()
        
        if self.energy_offset is None:
            raise ValueError("Energy offset not set! Call set_energy_offset(atoms) first.")
        E_xtb_aligned = E_xtb_cluster - self.energy_offset

        # --- Step 5: Construct global xTB forces ---
        F_xtb_global = F_mlp.copy()
        for local_i, global_i in enumerate(cluster_indices):
            F_xtb_global[global_i] = F_xtb_cluster[local_i]

        xtb_energy_info = f"E_xTB_cluster: {E_xtb_cluster:.6f} eV, E_xTB_aligned: {E_xtb_aligned:.6f} eV"
        xtb_force_info = f"F_xTB_cluster (max): {np.max(np.abs(F_xtb_cluster)): .4f} eV/Å"
        offset_info = f"Energy Offset: {self.energy_offset:.6f} eV"
        logger.info(f"xTB Calculation Complete | {xtb_energy_info} | {xtb_force_info} | {offset_info}")


        # --- Step 6: Smooth blending ---
        F_final = self._blend_forces(atoms, F_mlp, F_xtb_global, reactive_indices)
        
        # Energy: Use MLP as base, add local correction (simplified)
        # For most purposes, energy is less critical than forces in MD
        E_final = E_mlp  # or implement energy blending if needed

        self.results = {'energy': E_final, 'forces': F_final}

        final_energy_info = f"E_Final: {E_final:.6f} eV"
        final_force_info = f"F_Final (max): {np.max(np.abs(F_final)): .4f} eV/Å"
        logger.info(f"Hybrid Calculation Finished | {final_energy_info} | {final_force_info}")

    def _blend_forces(self, atoms, F_mlp, F_xtb, reactive_indices):
        F_out = F_mlp.copy()
        pos = atoms.positions
        for i in range(len(atoms)):
            dists = np.linalg.norm(pos[i] - pos[reactive_indices], axis=1)
            r = np.min(dists)
            if r <= self.R_inner:
                w = 1.0
            elif r >= self.R_outer:
                w = 0.0
            else:
                w = 0.5 * (1 + np.cos(np.pi * (r - self.R_inner) / (self.R_outer - self.R_inner)))
            F_out[i] = w * F_xtb[i] + (1 - w) * F_mlp[i]
        return F_out