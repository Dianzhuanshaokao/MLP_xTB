from ase.neighborlist import neighbor_list
import numpy as np

def analyze_reaction(atoms, verbose=False):
    """
    Unified reaction analyzer for Li-P-S-Cl systems.
    
    Returns:
        is_reactive (bool): Whether a reaction event occurred.
        reactive_indices (np.ndarray): Indices of atoms involved in the reaction.
    """
    symbols = np.array(atoms.get_chemical_symbols())
    positions = atoms.get_positions()
    
    # Build neighbor list once
    i_list, j_list, d_list = neighbor_list('ijd', atoms, cutoff=5.0)
    
    reactive_mask = np.zeros(len(atoms), dtype=bool)
    li_s_bond_count = 0
    p_s_broken = False
    cl_moved = False

    # 1. Li-S bond formation (reduction)
    for i, j, d in zip(i_list, j_list, d_list):
        if {symbols[i], symbols[j]} == {'Li', 'S'} and d < 2.4:
            reactive_mask[i] = reactive_mask[j] = True
            li_s_bond_count += 1

    # 2. P-S bond breaking (decomposition)
    for i, j, d in zip(i_list, j_list, d_list):
        if {symbols[i], symbols[j]} == {'P', 'S'} and d > 2.7:
            reactive_mask[i] = reactive_mask[j] = True
            p_s_broken = True

    # 3. Cl escape
    cl_indices = np.where(symbols == 'Cl')[0]
    if len(cl_indices) > 0:
        cl_idx = cl_indices[0]
        ps_indices = np.where((symbols == 'P') | (symbols == 'S'))[0]
        if len(ps_indices) > 0:
            cl_pos = positions[cl_idx]
            ps_positions = positions[ps_indices]
            min_dist = np.min(np.linalg.norm(ps_positions - cl_pos, axis=1))
            if min_dist > 4.0:
                reactive_mask[cl_idx] = True
                reactive_mask[ps_indices] = True
                cl_moved = True

    # Determine if reaction occurred
    is_reactive = (li_s_bond_count >= 3) or p_s_broken or cl_moved
    reactive_indices = np.where(reactive_mask)[0]

    if verbose and is_reactive:
        print(f"  Li-S bonds: {li_s_bond_count}, P-S broken: {p_s_broken}, Cl moved: {cl_moved}")
        print(f"  Reactive atoms: {len(reactive_indices)}")

    return is_reactive, reactive_indices