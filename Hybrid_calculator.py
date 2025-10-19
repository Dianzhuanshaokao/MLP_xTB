from ase import neighborlist
import numpy as np
from ase.calculators.calculator import Calculator, all_changes

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
        
        atoms_ref.calc = self.xtb_calc
        E_xtb = atoms_ref.get_potential_energy()
        
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

        # --- Step 2: If no reactive atoms, return MLP result ---
        if len(reactive_indices) == 0:
            self.results = {'energy': E_mlp, 'forces': F_mlp}
            return

        # --- Step 3: Build cluster (reactive atoms + buffer) ---
        cutoff = self.R_outer + 1.0
        nl = neighborlist(cutoffs=[cutoff/2]*len(atoms), skin=0.0, self_interaction=False, bothways=True)
        nl.update(atoms)
        
        cluster_mask = np.zeros(len(atoms), dtype=bool)
        for i in reactive_indices:
            indices, _ = nl.get_neighbors(i)
            cluster_mask[indices] = True
        cluster_mask[reactive_indices] = True
        cluster_indices = np.where(cluster_mask)[0]
        
        cluster_atoms = atoms[cluster_indices]
        cluster_atoms.calc = self.xtb_calc
        
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

        # --- Step 6: Smooth blending ---
        F_final = self._blend_forces(atoms, F_mlp, F_xtb_global, reactive_indices)
        
        # Energy: Use MLP as base, add local correction (simplified)
        # For most purposes, energy is less critical than forces in MD
        E_final = E_mlp  # or implement energy blending if needed

        self.results = {'energy': E_final, 'forces': F_final}

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

class HybridCalculator2:
    def __init__(self, mace_calc, xtb_calc, atoms, reactive_indices, R_inner=2.5, R_outer=4.5):
        self.mlp_calc = mace_calc
        self.xtb_calc = xtb_calc
        self.energy_offset = self.energy_alignment(atoms)
        self.R_inner = R_inner
        self.R_outer = R_outer
        self.cluster_atoms, self.cluster_indicies = self.cluster(atoms, reactive_indices)

    def cluster(self, atoms, reactive_indices):
        
        nl = neighborlist.NeighborList(self.R_outer + 1.0, skin=0.0, self_interaction=False).update(atoms)
        cluster_mask = np.zeros(len(atoms), dtype=bool)
        for i in reactive_indices:
            indices, _ = nl.get_neighbors(i)
            cluster_mask[indices] = True
        cluster_mask[reactive_indices] = True
        cluster_indices = np.where(cluster_mask)[0]
        
        cluster_atoms = atoms[cluster_indices]
        return cluster_atoms, cluster_indices
    
    def energy_alignment(self, atoms):
        atoms_ref = atoms.copy()
        atoms_ref.calc = self.mlp_calc
        E_mlp_ref = atoms_ref.get_potential_energy()

        atoms_ref_xtb = atoms_ref.copy()
        atoms_ref_xtb.calc = self.xtb_calc
        E_xtb_ref = atoms_ref_xtb.get_potential_energy()
        return E_xtb_ref - E_mlp_ref

    def get_potential_energy(self, atoms, reactive_indices):
        atoms.calc = self.mlp_calc
        E_mlp = atoms.get_potential_energy()

        if len(reactive_indices) == 0: return E_mlp

        E_xtb_cluster = self.cluster_atoms.get_potential_energy() - self.energy_offset
        E_final = self._blend_energy(atoms, E_mlp, E_xtb_cluster, reactive_indices)
        return E_final

    def get_forces(self, atoms, reactive_indices):
        atoms.calc = self.mlp_calc
        F_mlp = atoms.get_forces()

        if len(reactive_indices) == 0: return F_mlp
        self.cluster_atoms.calc = self.xtb_calc
        
        F_xtb_cluster = self.cluster_atoms.get_forces()
        F_xtb_global = F_mlp.copy()
        for local_i, global_i in enumerate(self.cluster_indices): F_xtb_global[global_i] = F_xtb_cluster[local_i]
        F_final = self._blend_forces(atoms, F_mlp, F_xtb_global, reactive_indices)

        return F_final

    def _blend_forces(self, atoms, F_mlp, F_xtb, reactive_indices):
        F_out = F_mlp.copy()
        pos = atoms.positions
        for i in range(len(atoms)):
            if len(reactive_indices) == 0:
                continue
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